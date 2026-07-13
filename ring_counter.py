"""
diffraction_ring_counter_bothaxis.py
====================================

回折リング（透過光の周りにできる同心楕円状のリング）の本数を、
アップロードした画像から自動でカウントするスクリプト。

★このバージョンの動作（両軸・多い方採用版 / 3 領域分割版）★
  - 楕円中心から長軸に沿って、+ 方向 / - 方向の【両側】それぞれ独立に
    片側スペクトルを取得し、それぞれで独立にピーク検出を行う。
  - + 側と - 側のピークをペアリング／マッチングする処理は一切行わない。
  - 検出本数が多かった側の本数を「リング数」として出力する。

★半径方向を 3 分割して解析する★
      inner  : ignore_center  <= r <  inner_radius   → ショルダー検出（2次微分）
      middle : inner_radius   <= r <  middle_radius  → ピークトップ検出（1次微分）
      outer  : middle_radius  <= r                   → ピークトップ検出（1次微分）
  middle と outer は検出【手法】は同じだが、パラメータ（最小距離・最小ピーク
  高さ・Savitzky-Golay 窓・山頂探索窓）は領域ごとに独立に設定できる。

★パラメータの変更箇所は Params データクラス（＋ argparse）の1箇所のみ。★
  検出ロジック・CSV 出力・グラフ描画のすべてが同じ Params を参照するため、
  平滑化窓・ピーク間隔・領域境界を変えても不整合が起きない。

処理の流れ
----------
STEP 0: 入力画像を平滑化する（Fiji の Process > Smooth = 3x3 平均フィルタ相当）。
        --presmooth N で N 回反復する。Fiji で事前に Smooth していない
        生画像をそのまま入力できる。
STEP 1: 回折リングを楕円近似し、楕円長軸の角度を求める。
STEP 2: 楕円長軸方向に沿って + 側 / - 側それぞれの片側プロファイルを取得する。
        --line-width N を指定すると、長軸に【垂直な方向】に N 本ぶん
        サンプリングして平均し、S/N を上げる。方位角方向にリングが一様で
        あることを利用するので、半径方向の分解能は落ちない。
        --width-mode arc なら楕円弧（リング）に沿って平均するため、
        幅を広げても外側のピークが鈍らない。
STEP 3: 各側の輝度プロファイルを 3 領域に分けてピーク位置を算出する。
        [inner ] 透過光の裾に紛れてリングが独立した山にならず "ショルダー"
                 形状になるため、2 回微分してその変曲点（2次微分の谷）を検出。
        [middle] Savitzky-Golay で平滑化・微分し、1 次微分のゼロ交差かつ
                 2 次微分が負（上に凸）の点をピークトップとして検出。
        [outer ] middle と同じ手法。パラメータのみ独立。
STEP 4: + 側 / - 側の検出本数を比較し、多い方をリング数として採用する。

出力
----
  <prefix>_annotated.png     : 両側の検出結果を重ねた画像（採用側を強調）
  <prefix>_smoothed.png      : STEP 0 で平滑化した画像（--presmooth 使用時のみ）
  <prefix>_plus_profile.png  : + 側プロファイルと検出位置の図
  <prefix>_minus_profile.png : - 側プロファイルと検出位置の図
  <prefix>_plus_data.csv     : + 側の生輝度・平滑化・微分・検出ピーク
  <prefix>_minus_data.csv    : - 側の同上
    （CSV 列: position_px, raw_brightness, smoothed_brightness, region,
              first_derivative, second_derivative, is_detected_peak, peak_type）

使い方
------
  python diffraction_ring_counter_bothaxis.py image.png

  # 3 領域の境界を指定
  python diffraction_ring_counter_bothaxis.py image.png \
        --inner-radius 60 --middle-radius 140

  # 領域ごとにパラメータを個別設定
  python diffraction_ring_counter_bothaxis.py image.png \
        --inner-distance 4  --inner-height 0.3 --inner-window 11 \
        --middle-distance 5 --middle-height 3.0 --middle-window 11 \
        --outer-distance 8  --outer-height 6.0 --outer-window 15

  # 長軸に垂直な方向に 9px 分を平均してプロファイルを取る（推奨）
  python diffraction_ring_counter_bothaxis.py image.png --line-width 9
  # Fiji の Line Width 相当（垂直な直線上で平均）
  python diffraction_ring_counter_bothaxis.py image.png --line-width 9 --width-mode normal
  # --angle 指定時に arc を使うには軸比も渡す
  python diffraction_ring_counter_bothaxis.py image.png --angle 41.8 --axis-ratio 0.65 --line-width 9

  # Fiji で 10 回 Smooth していた前処理をコード内で行う（生画像を入力）
  python diffraction_ring_counter_bothaxis.py raw_image.png --presmooth 10

  # 中心・長軸角度を手動指定
  python diffraction_ring_counter_bothaxis.py image.png --center 288 358 --angle 41.8

必要ライブラリ: numpy, pandas, opencv-python, scipy, matplotlib, pillow
  ※ HEIC/HEIF を扱う場合は追加で: pip install pillow-heif
"""

import argparse
import io
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import cv2
from scipy.signal import find_peaks, savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ======================================================================
# ★ パラメータ定義：数値を変えたいときはここ（と argparse の default）だけ ★
# ======================================================================
@dataclass
class RegionParams:
    """1 つの半径領域に適用する検出パラメータ一式。

    method:
      "shoulder" → 2 次微分の谷（変曲点）を検出（内側の強い裾の中の肩を拾う）
      "peaktop"  → 1 次微分のゼロ交差かつ 2 次微分が負の点を検出（独立した山）
    """
    name: str                      # "inner" / "middle" / "outer"
    method: str                    # "shoulder" or "peaktop"
    r_min: float                   # この領域の内側境界[px]（この値を含む）
    r_max: float                   # この領域の外側境界[px]（この値を含まない、inf 可）
    distance: float                # ピーク間の最小距離[px]
    height: float                  # 最小ピーク高さ（prominence）
    window: int                    # Savitzky-Golay 平滑化窓
    valley_guard: int = 3          # [shoulder 用] 谷底除外の近傍幅[サンプル]
    search_win: int = 4            # [peaktop 用] 山頂スナップの探索窓[サンプル]

    def mask(self, t):
        """この領域に属するサンプルの真偽マスクを返す。"""
        return (t >= self.r_min) & (t < self.r_max)


@dataclass
class Params:
    # --- 前処理（Fiji の Process > Smooth 相当）---
    presmooth: int = 15             # 3x3 平均フィルタの反復回数（0 = 適用しない）
    presmooth_ksize: int = 3       # 平均フィルタのカーネルサイズ（Fiji は 3 固定）
    presmooth_uint8: bool = False  # True で各反復ごとに 8bit 丸め（ImageJ 完全一致）

    # --- 領域境界（inner < middle < outer）---
    ignore_center: float = 10.0     # 中心（透過光そのもの）を除外する半径[px]
    inner_radius: float = 80.0     # inner / middle の境界[px]
    middle_radius: float = 800.0   # middle / outer の境界[px]

    # --- inner（ショルダー検出：2次微分の谷）---
    inner_distance: float = 4.0
    inner_height: float = 0.3
    inner_window: int = 7
    inner_valley_guard: int = 3

    # --- middle（ピークトップ検出：1次微分ゼロ交差）---
    middle_distance: float = 20.0
    middle_height: float = 10.0
    middle_window: int = 10
    middle_search_win: int = 4

    # --- outer（ピークトップ検出：1次微分ゼロ交差）---
    outer_distance: float = 70.0
    outer_height: float = 30.0
    outer_window: int = 15
    outer_search_win: int = 4

    # --- プロファイル取得（長軸に垂直な方向の平均）---
    line_width: int = 5            # 法線方向にサンプリングする本数[px]（1=単一ライン）
    width_mode: str = "normal"        # "arc"（楕円弧に沿う）or "normal"（垂直な直線）

    # --- 共通 ---
    polyorder: int = 3             # Savitzky-Golay 多項式次数
    boundary_guard: float = 5.0    # 領域境界での二重検出除去距離[px]
    min_samples: int = 10          # 各領域を処理するのに必要な最小サンプル数
    step: float = 1.0              # プロファイルのサンプリング刻み[px]

    def sanitize_window(self, window, n_samples):
        """窓長を「奇数」「polyorder より大きい」「データ長以下」に丸める。"""
        w = int(min(window, n_samples - 1))
        if w % 2 == 0:
            w -= 1
        w = max(w, self.polyorder + 2 if (self.polyorder + 2) % 2 else self.polyorder + 3)
        return w

    def regions(self):
        """3 領域の定義を内側から順に返す（検出・CSV・描画で共用する唯一の定義）。"""
        return [
            RegionParams("inner", "shoulder",
                         self.ignore_center, self.inner_radius,
                         self.inner_distance, self.inner_height, self.inner_window,
                         valley_guard=self.inner_valley_guard),
            RegionParams("middle", "peaktop",
                         self.inner_radius, self.middle_radius,
                         self.middle_distance, self.middle_height, self.middle_window,
                         search_win=self.middle_search_win),
            RegionParams("outer", "peaktop",
                         self.middle_radius, np.inf,
                         self.outer_distance, self.outer_height, self.outer_window,
                         search_win=self.outer_search_win),
        ]

    def validate(self):
        if not (self.ignore_center < self.inner_radius < self.middle_radius):
            raise ValueError(
                "領域境界は ignore_center < inner_radius < middle_radius を"
                f"満たす必要があります（現在: {self.ignore_center} < "
                f"{self.inner_radius} < {self.middle_radius}）")
        if self.line_width < 1:
            raise ValueError(f"line_width は 1 以上（現在: {self.line_width}）")
        if self.width_mode not in ("arc", "normal"):
            raise ValueError(f"width_mode は arc / normal（現在: {self.width_mode}）")


# ======================================================================
# 画像読み込み（PNG / JPEG / TIFF / BMP / WEBP / HEIC / HEIF 対応）
# ======================================================================
def load_image_bgr(path):
    """画像ファイルを OpenCV の BGR 配列（uint8, 3ch）として読み込む。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"ファイルが存在しません: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext in (".heic", ".heif"):
        try:
            from PIL import Image, ImageOps
        except ImportError:
            raise ImportError(
                "HEIC/HEIF を読むには Pillow が必要です: pip install pillow pillow-heif")
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            pass
        try:
            pil_img = Image.open(path)
            pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        except Exception as e:
            raise RuntimeError(
                "HEIC/HEIF を読み込めませんでした。pillow-heif の導入が必要かもしれません "
                f"（pip install pillow-heif）: {e}")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return img

    try:
        from PIL import Image, ImageOps
        pil_img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        pass

    raise ValueError(
        f"画像を読み込めません（未対応の形式か、破損している可能性があります）: {path}")


# ======================================================================
# STEP 0 : 画像の平滑化（Fiji の Process > Smooth 相当）
# ======================================================================
def apply_fiji_smooth(gray, p: Params):
    """Fiji の Process > Smooth（3x3 平均フィルタ 1 回）を p.presmooth 回反復する。

    Fiji/ImageJ の Smooth は 3x3 の等重み平均（mean）フィルタで、
    画像の端は「端の画素を複製して外挿」して処理される。
    ここでは cv2.blur + BORDER_REPLICATE で同じ挙動を再現する。

    p.presmooth_uint8 = False（既定）:
        float32 のまま反復するので丸め誤差が蓄積せず、
        微分に基づくピーク検出には有利。
    p.presmooth_uint8 = True:
        各反復ごとに 8bit へ丸めるため ImageJ の結果とほぼ一致する。

    返り値は float32 の 2 次元配列。
    """
    if p.presmooth <= 0:
        return gray.astype(np.float32)

    k = p.presmooth_ksize
    if k % 2 == 0:
        k += 1  # 偶数カーネルは中心がずれるので奇数に丸める

    out = gray.astype(np.float32)
    for _ in range(p.presmooth):
        out = cv2.blur(out, (k, k), borderType=cv2.BORDER_REPLICATE)
        if p.presmooth_uint8:
            out = np.clip(np.round(out), 0, 255)
    return out


def to_uint8(gray_f):
    """Canny/GaussianBlur など uint8 前提の OpenCV 関数へ渡すための変換。"""
    return np.clip(gray_f, 0, 255).astype(np.uint8)


# ======================================================================
# STEP 1 : 楕円近似 → 透過光中心（固定）と長軸角度を求める
# ======================================================================
def find_transmitted_center(gray, core_fraction=0.5):
    """透過光スポットの中心を輝度加重重心でサブピクセル精度に求める。"""
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    _, _, _, max_loc = cv2.minMaxLoc(blurred)
    x0, y0 = max_loc
    h, w = gray.shape
    win = max(6, int(round(min(h, w) * 0.04)))
    x1, x2 = max(0, x0 - win), min(w, x0 + win + 1)
    y1, y2 = max(0, y0 - win), min(h, y0 + win + 1)
    patch = gray[y1:y2, x1:x2].astype(np.float64)
    weights = np.where(patch >= patch.max() * core_fraction, patch, 0.0)
    ys, xs = np.indices(patch.shape)
    total = weights.sum()
    if total <= 0:
        return float(x0), float(y0)
    cx = x1 + float((xs * weights).sum() / total)
    cy = y1 + float((ys * weights).sum() / total)
    return cx, cy


def fit_ellipse_fixed_center(points, center):
    """中心固定の楕円 A*dx^2 + B*dx*dy + C*dy^2 = 1 を最小二乗フィットする。"""
    cx, cy = center
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    M = np.column_stack([dx ** 2, dx * dy, dy ** 2])
    b = np.ones(len(dx))
    try:
        sol, *_ = np.linalg.lstsq(M, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    A, B, C = sol
    Mat = np.array([[A, B / 2.0], [B / 2.0, C]])
    try:
        eigvals, eigvecs = np.linalg.eigh(Mat)
    except np.linalg.LinAlgError:
        return None
    if np.any(eigvals <= 0):
        return None
    axes = 1.0 / np.sqrt(eigvals)
    order = np.argsort(axes)[::-1]
    axes_sorted, vecs_sorted = axes[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(vecs_sorted[1, 0], vecs_sorted[0, 0])) % 180.0
    return axes_sorted[0], axes_sorted[1], angle


def detect_major_axis_angle(gray, center, min_contour_len=40, residual_thresh=0.08):
    """リング輪郭に楕円をフィットし、長軸角度と軸比をロバストに求める。"""
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
    med = np.median(blurred)
    edges = cv2.Canny(blurred, int(max(0, 0.66 * med)), int(min(255, 1.33 * med)))
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    fits = []
    for c in contours:
        if len(c) < min_contour_len:
            continue
        pts = c.reshape(-1, 2).astype(np.float64)
        result = fit_ellipse_fixed_center(pts, center)
        if result is None:
            continue
        major, minor, angle = result
        if major <= 0 or minor <= 0:
            continue
        cx, cy = center
        dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
        theta = np.deg2rad(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        xr = dx * cos_t + dy * sin_t
        yr = -dx * sin_t + dy * cos_t
        norm_r = np.sqrt((xr / major) ** 2 + (yr / minor) ** 2)
        if np.mean(np.abs(norm_r - 1.0)) < residual_thresh:
            fits.append({"major": major, "minor": minor, "angle": angle})

    if not fits:
        return None, None

    angles = np.array([f["angle"] for f in fits])
    double_rad = np.deg2rad(angles * 2)
    mean_vec = np.array([np.mean(np.cos(double_rad)), np.mean(np.sin(double_rad))])
    angle_deg = (np.rad2deg(np.arctan2(mean_vec[1], mean_vec[0])) / 2.0) % 180.0
    axis_ratio = float(np.median([f["minor"] / f["major"] for f in fits]))
    return angle_deg, axis_ratio


# ======================================================================
# STEP 2 : 楕円長軸方向「片側のみ」の輝度プロファイルを取得
# ======================================================================
def _remap_nan(gray_f, xs, ys):
    """(xs, ys) をバイリニア補間で読む。画像外は NaN にして平均から除外する。"""
    return cv2.remap(gray_f,
                     xs.astype(np.float32).reshape(1, -1),
                     ys.astype(np.float32).reshape(1, -1),
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT,
                     borderValue=float("nan")).reshape(-1)


def sample_halfline_profile(gray, center, angle_deg, side, p: Params, axis_ratio=None):
    """中心を起点に angle_deg 方向（plus）または逆方向（minus）へ伸びる
    半直線に沿って輝度プロファイルを取得する。

    長軸に【垂直な方向】に p.line_width 本ぶんサンプリングして平均する。
    （line_width=1 なら従来どおり単一ラインで、結果は完全に不変）

    p.width_mode:
      "normal" … 長軸に垂直な直線上を等間隔にオフセットして平均する。
                 Fiji の Line Width を上げた Plot Profile に相当。
                 実装が単純だが、半径 r が大きいところではリング（楕円弧）を
                 斜めに横切るため、幅を広げるとピークが鈍る。
      "arc"    … 楕円の「等半径パラメータ」曲線（＝リングそのもの）に
                 沿ってオフセットして平均する。リングに沿って平均するので
                 幅を広げても半径方向の分解能が落ちにくい。軸比が必要。

    垂直方向の平均は、リングが方位角方向にほぼ一様であることを利用して
    S/N を稼ぐ操作であり、半径方向（＝プロファイル方向）の分解能を
    犠牲にしない点が --presmooth（等方的な 2 次元平滑化）との違いである。

    返り値
      t      : 中心からの距離[px]（0 以上の昇順）
      values : その位置の輝度（垂直方向に平均済み）
    """
    h, w = gray.shape
    cx, cy = center
    theta = np.deg2rad(angle_deg)
    sign = 1.0 if side == "plus" else -1.0

    # 長軸方向（プロファイルが伸びる向き）と、その法線方向
    ux, uy = sign * np.cos(theta), sign * np.sin(theta)
    nx, ny = -np.sin(theta), np.cos(theta)

    # 半直線が画像内に収まる最大距離
    t = 0.0
    while True:
        x, y = cx + (t + p.step) * ux, cy + (t + p.step) * uy
        if not (0 <= x < w - 1 and 0 <= y < h - 1):
            break
        t += p.step
    t_vals = np.arange(0.0, t, p.step)

    gray_f = gray.astype(np.float32)

    # line_width=1 は単一ライン（従来動作）
    width = max(1, int(p.line_width))
    if width == 1:
        values = _remap_nan(gray_f, cx + t_vals * ux, cy + t_vals * uy)
        return t_vals, np.nan_to_num(values, nan=0.0)

    half = (width - 1) / 2.0
    offsets = np.linspace(-half, half, width)  # 法線方向のオフセット[px]

    mode = p.width_mode
    if mode == "arc" and not axis_ratio:
        mode = "normal"  # 軸比が不明なら直線オフセットにフォールバック

    stack = np.full((width, len(t_vals)), np.nan, dtype=np.float64)
    for k, o in enumerate(offsets):
        if mode == "arc":
            # 楕円の等半径パラメータ曲線上を動く。
            # 長軸半径 a, 短軸半径 b, 軸比 q=b/a のとき
            #     (xr/a)^2 + (yr/b)^2 = (t/a)^2
            # が同一リング。法線オフセット yr=o を与えると xr は次のように縮む:
            #     xr = sqrt(t^2 - (o/q)^2)
            shrink = (o / axis_ratio) ** 2
            xr = np.sqrt(np.maximum(t_vals ** 2 - shrink, 0.0))
            valid = t_vals ** 2 >= shrink  # 中心近傍はこのリングに乗れない
            row = _remap_nan(gray_f, cx + xr * ux + o * nx, cy + xr * uy + o * ny)
            row[~valid] = np.nan
        else:
            # 長軸に垂直な直線上を平行移動
            row = _remap_nan(gray_f,
                             cx + t_vals * ux + o * nx,
                             cy + t_vals * uy + o * ny)
        stack[k] = row

    # 画像外／曲線外のサンプルを除いて平均（全滅した列は 0 埋め）
    with np.errstate(invalid="ignore"):
        values = np.nanmean(stack, axis=0)
    return t_vals, np.nan_to_num(values, nan=0.0)


# ======================================================================
# 平滑化・微分の共通ユーティリティ（検出・CSV・描画がすべてこれを使う）
# ======================================================================
def smooth_and_derivatives(values, window, p: Params, derivs=(0, 1, 2)):
    """Params の polyorder と丸め規則で平滑化・微分をまとめて返す。"""
    w = p.sanitize_window(window, len(values))
    return tuple(savgol_filter(values, window_length=w, polyorder=p.polyorder, deriv=d)
                 for d in derivs)


# ======================================================================
# STEP 3 : ピーク検出
#          shoulder 法（inner） / peaktop 法（middle, outer）
# ======================================================================
def detect_shoulder_peaks(t, values, rp: RegionParams, p: Params):
    """[shoulder 法] 2 次微分の谷（上に凸な変曲点）を検出し、谷底は除外する。

    透過光の強い裾に紛れてリングが独立した山にならず、"ショルダー" 形状に
    なる領域で用いる。真のピークもショルダーも共に 2 次微分の谷になる。
    """
    v_smooth, d2 = smooth_and_derivatives(values, rp.window, p, derivs=(0, 2))
    troughs, _ = find_peaks(-d2, distance=rp.distance, prominence=rp.height)

    kept = []
    for i in troughs:
        lo = max(0, i - rp.valley_guard)
        hi = min(len(v_smooth), i + rp.valley_guard + 1)
        if v_smooth[i] <= v_smooth[lo:hi].min() + 1e-9:
            continue  # 谷底は山ではないので除外
        kept.append(i)
    return t[np.array(kept, dtype=int)] if kept else np.array([])


def detect_peak_tops(t, values, rp: RegionParams, p: Params):
    """[peaktop 法] 1 次微分が + → - に交差し、かつ 2 次微分が負の点を
    ピークトップとして検出する（サブピクセル補間つき）。

    middle 領域と outer 領域で共通に用いる。適用するパラメータ
    （distance / height / window / search_win）は rp から取るため、
    領域ごとに独立に設定できる。
    """
    v_smooth, d1, d2 = smooth_and_derivatives(values, rp.window, p, derivs=(0, 1, 2))

    sign = np.sign(d1)
    cross_idx = np.where((sign[:-1] > 0) & (sign[1:] <= 0))[0]

    candidates_t, candidates_v = [], []
    for i in cross_idx:
        if d2[i] >= 0:
            continue  # 下に凸（谷）は除外、上に凸のみ採用
        d1_0, d1_1 = d1[i], d1[i + 1]
        t0, t1 = t[i], t[i + 1]
        t_cross = t0 if d1_1 == d1_0 else t0 + (d1_0 / (d1_0 - d1_1)) * (t1 - t0)
        candidates_t.append(t_cross)
        candidates_v.append(np.interp(t_cross, [t0, t1], [v_smooth[i], v_smooth[i + 1]]))

    if not candidates_t:
        return np.array([])

    order = np.argsort(candidates_t)
    candidates_t = np.array(candidates_t)[order]
    candidates_v = np.array(candidates_v)[order]

    # 各候補の高さ（prominence）：隣接候補間の最小値からの高さ
    proms = []
    for k in range(len(candidates_t)):
        left = candidates_t[k - 1] if k > 0 else t[0]
        right = candidates_t[k + 1] if k < len(candidates_t) - 1 else t[-1]
        mask = (t >= left) & (t <= right)
        local_min = v_smooth[mask].min() if mask.any() else candidates_v[k]
        proms.append(candidates_v[k] - local_min)
    keep = np.array(proms) >= rp.height
    candidates_t, candidates_v = candidates_t[keep], candidates_v[keep]

    # 最小距離を強制：近すぎる候補は高い方を残す
    accepted = []
    for idx in np.argsort(-candidates_v):
        tc = candidates_t[idx]
        if all(abs(tc - a) >= rp.distance for a in accepted):
            accepted.append(tc)

    # 平滑化曲線の実際の山頂へスナップ＋放物線フィットでサブピクセル化
    refined = []
    for tc in accepted:
        i = int(np.argmin(np.abs(t - tc)))
        lo = max(0, i - rp.search_win)
        hi = min(len(v_smooth), i + rp.search_win + 1)
        rel_top = int(np.argmax(v_smooth[lo:hi]))
        top_i = lo + rel_top
        if rel_top == 0 or rel_top == (hi - lo - 1):
            continue  # 近傍が単調＝真の山頂が無い → 偽ピーク
        if 0 < top_i < len(t) - 1:
            y0, y1, y2 = v_smooth[top_i - 1], v_smooth[top_i], v_smooth[top_i + 1]
            denom = y0 - 2 * y1 + y2
            if denom < 0:
                delta = 0.5 * (y0 - y2) / denom
                step_local = t[top_i + 1] - t[top_i]
                refined.append(t[top_i] + delta * step_local
                               if abs(delta) <= 1.0 else t[top_i])
            else:
                refined.append(t[top_i])
        else:
            refined.append(t[top_i])

    final = []
    for tc in sorted(set(np.round(refined, 3))):
        if all(abs(tc - a) >= rp.distance for a in final):
            final.append(tc)
    return np.array(final)


def detect_in_region(t, values, rp: RegionParams, p: Params):
    """1 領域に対し、その領域の method に応じた検出器を適用する。"""
    mask = rp.mask(t)
    if mask.sum() <= p.min_samples:
        return np.array([])
    if rp.method == "shoulder":
        return detect_shoulder_peaks(t[mask], values[mask], rp, p)
    elif rp.method == "peaktop":
        return detect_peak_tops(t[mask], values[mask], rp, p)
    raise ValueError(f"未知の検出手法: {rp.method}")


def detect_hybrid_oneside(t, values, p: Params):
    """片側プロファイルを 3 領域に分けて検出し、統合する。

    返り値: (検出位置[px, 昇順], 領域名の配列[同順])
    """
    regions = p.regions()
    per_region = [detect_in_region(t, values, rp, p) for rp in regions]

    # 領域境界で隣接領域が同じリングを二重検出したら、外側の検出を優先し
    # 内側の重複を捨てる（外側ほど山が明瞭でピーク位置が信頼できるため）
    for i in range(len(per_region) - 1):
        inner_pk, outer_pk = per_region[i], per_region[i + 1]
        if len(inner_pk) and len(outer_pk):
            keep = np.array([np.min(np.abs(outer_pk - q)) > p.boundary_guard
                             for q in inner_pk])
            per_region[i] = inner_pk[keep]

    vals = np.concatenate(per_region) if per_region else np.array([])
    names = np.concatenate([np.full(len(pk), rp.name, dtype=object)
                            for pk, rp in zip(per_region, regions)]) \
        if per_region else np.array([], dtype=object)

    order = np.argsort(vals)
    return vals[order], names[order]


# ======================================================================
# 片側の結果を図・CSV に保存
# ======================================================================
# 領域ごとの表示色（描画と凡例で共用）
REGION_COLORS = {"inner": "orange", "middle": "royalblue", "outer": "green"}
REGION_BGR = {"inner": (0, 165, 255), "middle": (255, 128, 0), "outer": (0, 255, 0)}
REGION_BGR_DIM = {"inner": (0, 110, 170), "middle": (170, 90, 0), "outer": (0, 170, 0)}


def draw_annotated(img, gray, center, angle_deg, results, adopted_side, n_rings,
                   p: Params):
    """検出結果を重ねた BGR 画像を返す（CLI と表示用で共有）。"""
    annotated = img.copy()
    cx, cy = center
    theta = np.deg2rad(angle_deg)
    L = max(gray.shape)
    for side in ("plus", "minus"):
        sign = 1.0 if side == "plus" else -1.0
        dxs, dys = sign * np.cos(theta), sign * np.sin(theta)
        line_col = (255, 0, 255) if side == adopted_side else (160, 90, 160)
        cv2.line(annotated, (int(cx), int(cy)),
                 (int(cx + L * dxs), int(cy + L * dys)), line_col, 1, cv2.LINE_AA)
        # 領域境界を長軸上に小さな刻みで示す
        for r_bound in (p.inner_radius, p.middle_radius):
            bx, by = int(cx + r_bound * dxs), int(cy + r_bound * dys)
            cv2.drawMarker(annotated, (bx, by), (200, 0, 200),
                           markerType=cv2.MARKER_TILTED_CROSS,
                           markerSize=8, thickness=1)
        for rad, rname in zip(results[side]["peaks"], results[side]["regions"]):
            x1, y1 = int(cx + rad * dxs), int(cy + rad * dys)
            if side == adopted_side:
                cv2.circle(annotated, (x1, y1), 4, REGION_BGR[rname], -1)
            else:
                cv2.circle(annotated, (x1, y1), 3, REGION_BGR_DIM[rname], 1, cv2.LINE_AA)
    cv2.drawMarker(annotated, (int(cx), int(cy)), (0, 0, 255),
                   markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
    cv2.putText(annotated, f"Rings: {n_rings} ({adopted_side})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    return annotated



def save_side_outputs(side, t, values, peaks, peak_regions, p: Params,
                      out_prefix, adopted, return_png=False, dpi=150,
                      write_csv=True):
    """1 つの側（plus/minus）についてプロファイル図と CSV を保存する。
    平滑化は検出ロジックと完全に同じ窓長・次数を使う。"""
    regions = p.regions()
    n_rings = len(peaks)

    # --- プロファイル図（領域ごとに検出と同一の窓で平滑化）---
    curve = np.full_like(values, np.nan, dtype=float)
    for rp in regions:
        m = rp.mask(t)
        if m.sum() > p.min_samples:
            curve[m] = smooth_and_derivatives(values[m], rp.window, p, derivs=(0,))[0]

    plt.figure(figsize=(10, 4.2))
    plt.plot(t, curve, color="steelblue", linewidth=1,
             label=f"Smoothed (win: in={p.inner_window}, mid={p.middle_window}, "
                   f"out={p.outer_window}, polyorder={p.polyorder})")

    # 領域境界の縦線
    for r_bound, lbl in ((p.inner_radius, "inner/middle"),
                         (p.middle_radius, "middle/outer")):
        plt.axvline(r_bound, color="purple", linestyle=":", linewidth=1)
    plt.plot([], [], color="purple", linestyle=":", label="region boundaries")

    # 領域の背景を薄く塗り分ける
    finite_max = np.nanmax(t) if len(t) else 1.0
    for rp in regions:
        lo, hi = rp.r_min, min(rp.r_max, finite_max)
        if hi > lo:
            plt.axvspan(lo, hi, color=REGION_COLORS[rp.name], alpha=0.06)

    # 検出ピーク
    for r, rname in zip(peaks, peak_regions):
        plt.axvline(r, color=REGION_COLORS[rname], linestyle="--", linewidth=0.9)
    counts = {rp.name: int(np.sum(peak_regions == rp.name)) for rp in regions}
    for rp in regions:
        method = "shoulder" if rp.method == "shoulder" else "peak top"
        plt.plot([], [], color=REGION_COLORS[rp.name], linestyle="--",
                 label=f"{rp.name} ({method}): {counts[rp.name]}")

    plt.xlabel(f"Distance from center along major axis (px, side={side})")
    plt.ylabel("Brightness (smoothed grayscale)")
    plt.title(f"side={side}: {n_rings} rings detected"
              f"{'  [ADOPTED]' if adopted else ''}  "
              f"(inner={counts['inner']}, middle={counts['middle']}, "
              f"outer={counts['outer']})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    if return_png:
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=dpi)
        plt.close()
        buf.seek(0)
        return buf.getvalue()
    plt.savefig(f"{out_prefix}_{side}_profile.png", dpi=150)
    plt.close()

    # --- CSV ---
    rows = []
    for rp in regions:
        m = rp.mask(t)
        if m.sum() <= p.min_samples:
            continue
        if rp.method == "shoulder":
            sm, d2 = smooth_and_derivatives(values[m], rp.window, p, (0, 2))
            d1 = np.full(len(sm), np.nan)   # shoulder 法は 1 次微分を使わない
        else:
            sm, d1, d2 = smooth_and_derivatives(values[m], rp.window, p, (0, 1, 2))
        for tt, vv, sv, a, b in zip(t[m], values[m], sm, d1, d2):
            rows.append({"position_px": tt, "raw_brightness": vv,
                         "smoothed_brightness": sv, "region": rp.name,
                         "first_derivative": a, "second_derivative": b,
                         "is_detected_peak": False, "peak_type": ""})

    df = pd.DataFrame(rows).sort_values("position_px").reset_index(drop=True)
    method_of = {rp.name: rp.method for rp in regions}
    for r, rname in zip(peaks, peak_regions):
        idx = (df["position_px"] - r).abs().idxmin()
        df.at[idx, "is_detected_peak"] = True
        df.at[idx, "peak_type"] = method_of[rname]
    if write_csv:
        df.to_csv(f"{out_prefix}_{side}_data.csv", index=False)


# ======================================================================
# メイン処理
# ======================================================================
def build_params(args) -> Params:
    """argparse の結果を Params に詰め替える（唯一の変換点）。"""
    return Params(
        line_width=args.line_width,
        width_mode=args.width_mode,
        presmooth=args.presmooth,
        presmooth_ksize=args.presmooth_ksize,
        presmooth_uint8=args.presmooth_uint8,
        ignore_center=args.ignore_center,
        inner_radius=args.inner_radius,
        middle_radius=args.middle_radius,
        inner_distance=args.inner_distance,
        inner_height=args.inner_height,
        inner_window=args.inner_window,
        inner_valley_guard=args.inner_valley_guard,
        middle_distance=args.middle_distance,
        middle_height=args.middle_height,
        middle_window=args.middle_window,
        middle_search_win=args.middle_search_win,
        outer_distance=args.outer_distance,
        outer_height=args.outer_height,
        outer_window=args.outer_window,
        outer_search_win=args.outer_search_win,
        polyorder=args.polyorder,
        boundary_guard=args.boundary_guard,
    )


def main():
    d = Params()  # デフォルト値の供給元（argparse と二重管理しない）
    parser = argparse.ArgumentParser(
        description="回折リングを楕円近似し、長軸の【両側】それぞれ独立に "
                    "3 領域（inner=2次微分の変曲点 / middle・outer=ピークトップ）に "
                    "分けてピークを数え、検出本数が多かった側をリング数として出力する。")
    parser.add_argument("image", help="入力画像のパス")
    parser.add_argument("--center", nargs=2, type=float, metavar=("X", "Y"),
                        help="透過光中心を手動指定（省略時は自動検出）")
    parser.add_argument("--angle", type=float, default=None,
                        help="長軸角度[deg]を手動指定（省略時は自動フィット）")

    g = parser.add_argument_group("プロファイル取得（長軸に垂直な方向の平均）")
    g.add_argument("--line-width", type=int, default=d.line_width,
                   help=f"長軸に垂直な方向にサンプリングして平均する本数[px]。"
                        f"1 なら単一ライン（default: {d.line_width}）")
    g.add_argument("--width-mode", choices=("arc", "normal"), default=d.width_mode,
                   help=f"arc=楕円弧（リング）に沿って平均し半径分解能を保つ / "
                        f"normal=長軸に垂直な直線上で平均（Fiji の Line Width 相当）"
                        f"（default: {d.width_mode}）")
    g.add_argument("--axis-ratio", type=float, default=None,
                   help="軸比 b/a を手動指定。--angle を指定して自動フィットを"
                        "行わない場合、arc モードにはこの値が必要")

    g = parser.add_argument_group("前処理（Fiji の Process > Smooth 相当）")
    g.add_argument("--presmooth", type=int, default=d.presmooth,
                   help=f"3x3 平均フィルタの反復回数。Fiji で Smooth した回数と"
                        f"同じ値を指定する（default: {d.presmooth} = 適用しない）")
    g.add_argument("--presmooth-ksize", type=int, default=d.presmooth_ksize,
                   help=f"平均フィルタのカーネルサイズ。Fiji は 3 固定"
                        f"（default: {d.presmooth_ksize}）")
    g.add_argument("--presmooth-uint8", action="store_true", default=d.presmooth_uint8,
                   help="各反復ごとに 8bit へ丸める（ImageJ の結果に厳密に合わせる）")

    g = parser.add_argument_group("領域境界（ignore_center < inner_radius < middle_radius）")
    g.add_argument("--ignore-center", type=float, default=d.ignore_center,
                   help=f"中心（透過光）を除外する半径[px]（default: {d.ignore_center}）")
    g.add_argument("--inner-radius", type=float, default=d.inner_radius,
                   help=f"inner / middle の境界半径[px]（default: {d.inner_radius}）")
    g.add_argument("--middle-radius", type=float, default=d.middle_radius,
                   help=f"middle / outer の境界半径[px]（default: {d.middle_radius}）")

    g = parser.add_argument_group("inner 領域（ショルダー検出：2次微分の谷）")
    g.add_argument("--inner-distance", type=float, default=d.inner_distance,
                   help=f"検出点間の最小距離[px]（default: {d.inner_distance}）")
    g.add_argument("--inner-height", type=float, default=d.inner_height,
                   help=f"最小ピーク高さ（2次微分の顕著さ）（default: {d.inner_height}）")
    g.add_argument("--inner-window", type=int, default=d.inner_window,
                   help=f"Savitzky-Golay 平滑化窓（default: {d.inner_window}）")
    g.add_argument("--inner-valley-guard", type=int, default=d.inner_valley_guard,
                   help=f"谷底除外の近傍幅（default: {d.inner_valley_guard}）")

    g = parser.add_argument_group("middle 領域（ピークトップ検出：1次微分ゼロ交差）")
    g.add_argument("--middle-distance", type=float, default=d.middle_distance,
                   help=f"ピークトップ間の最小距離[px]（default: {d.middle_distance}）")
    g.add_argument("--middle-height", type=float, default=d.middle_height,
                   help=f"最小ピーク高さ（生輝度スケール）（default: {d.middle_height}）")
    g.add_argument("--middle-window", type=int, default=d.middle_window,
                   help=f"Savitzky-Golay 平滑化窓（default: {d.middle_window}）")
    g.add_argument("--middle-search-win", type=int, default=d.middle_search_win,
                   help=f"山頂スナップの探索窓（default: {d.middle_search_win}）")

    g = parser.add_argument_group("outer 領域（ピークトップ検出：1次微分ゼロ交差）")
    g.add_argument("--outer-distance", type=float, default=d.outer_distance,
                   help=f"ピークトップ間の最小距離[px]（default: {d.outer_distance}）")
    g.add_argument("--outer-height", type=float, default=d.outer_height,
                   help=f"最小ピーク高さ（生輝度スケール）（default: {d.outer_height}）")
    g.add_argument("--outer-window", type=int, default=d.outer_window,
                   help=f"Savitzky-Golay 平滑化窓（default: {d.outer_window}）")
    g.add_argument("--outer-search-win", type=int, default=d.outer_search_win,
                   help=f"山頂スナップの探索窓（default: {d.outer_search_win}）")

    g = parser.add_argument_group("共通")
    g.add_argument("--polyorder", type=int, default=d.polyorder,
                   help=f"Savitzky-Golay 多項式次数（default: {d.polyorder}）")
    g.add_argument("--boundary-guard", type=float, default=d.boundary_guard,
                   help=f"領域境界での二重検出除去距離[px]（default: {d.boundary_guard}）")
    parser.add_argument("--out-prefix", default=None,
                        help="出力ファイルの接頭辞（省略時は入力名から生成）")
    args = parser.parse_args()

    p = build_params(args)
    try:
        p.validate()
    except ValueError as e:
        sys.exit(f"パラメータエラー: {e}")

    try:
        img = load_image_bgr(args.image)
    except Exception as e:
        sys.exit(f"画像を読み込めません: {e}")
    gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out_prefix = args.out_prefix or (args.image.rsplit(".", 1)[0] + "_rings_bothaxis")

    # --- STEP 0: 画像の平滑化（Fiji の Smooth 相当）---
    gray = apply_fiji_smooth(gray_raw, p)          # 解析用（float32）
    gray_u8 = to_uint8(gray)                       # 楕円フィット用（Canny は uint8 必須）
    if p.presmooth > 0:
        mode = "8bit丸めあり" if p.presmooth_uint8 else "float32保持"
        print(f"[STEP0] Smooth（{p.presmooth_ksize}x{p.presmooth_ksize} 平均）を "
              f"{p.presmooth} 回適用（{mode}）")
    else:
        print("[STEP0] Smooth なし（入力画像をそのまま使用）")

    # --- STEP 1: 楕円近似 → 中心・長軸角度 ---
    center = tuple(args.center) if args.center else find_transmitted_center(gray_u8)
    print(f"[STEP1] 透過光中心（固定）: ({center[0]:.2f}, {center[1]:.2f})")

    if args.angle is not None:
        angle_deg, axis_ratio = args.angle, None
    else:
        angle_deg, axis_ratio = detect_major_axis_angle(gray_u8, center)
        if angle_deg is None:
            print("        楕円角度を自動検出できず、0度（水平）を使用します。")
            angle_deg = 0.0
    if args.axis_ratio is not None:
        axis_ratio = args.axis_ratio  # 手動指定が最優先
    print(f"[STEP1] 長軸角度: {angle_deg:.1f} deg" +
          (f"  (軸比 b/a={axis_ratio:.3f})" if axis_ratio else "  (軸比 不明)"))

    # --- プロファイル取得法の確定（arc は軸比が必須）---
    eff_mode = p.width_mode
    if p.line_width > 1 and eff_mode == "arc" and not axis_ratio:
        eff_mode = "normal"
        print("        [警告] 軸比が不明なため arc モードを使えません。"
              "normal モードにフォールバックします "
              "（--axis-ratio で軸比を指定してください）。")
    if p.line_width > 1:
        print(f"[STEP2] 長軸に垂直な方向に {p.line_width}px 分を平均"
              f"（mode={eff_mode}）")
    else:
        print("[STEP2] 単一ライン（幅1px）のプロファイルを使用")

    # --- STEP 2 & 3: 両側それぞれ独立に取得・検出 ---
    print(f"[STEP3] 領域境界: [{p.ignore_center:.0f}, {p.inner_radius:.0f}) inner | "
          f"[{p.inner_radius:.0f}, {p.middle_radius:.0f}) middle | "
          f"[{p.middle_radius:.0f}, inf) outer   (polyorder={p.polyorder})")
    for rp in p.regions():
        print(f"        {rp.name:6s} [{rp.method:8s}] "
              f"距離={rp.distance}, 高さ={rp.height}, 窓={rp.window}")

    # arc へフォールバックした場合は axis_ratio=None を渡して normal を選ばせる
    ratio_arg = axis_ratio if eff_mode == "arc" else None

    results = {}
    for side in ("plus", "minus"):
        t, values = sample_halfline_profile(gray, center, angle_deg, side, p,
                                            axis_ratio=ratio_arg)
        peaks, peak_regions = detect_hybrid_oneside(t, values, p)
        results[side] = {"t": t, "values": values,
                         "peaks": peaks, "regions": peak_regions}
        counts = {rp.name: int(np.sum(peak_regions == rp.name)) for rp in p.regions()}
        print(f"[STEP2/3] side={side:5s}: 長さ {len(values)}px, 検出 {len(peaks)} 本 "
              f"(inner {counts['inner']} / middle {counts['middle']} / "
              f"outer {counts['outer']})")

    # --- STEP 4: 多い方を採用（マッチングなし）---
    n_plus, n_minus = len(results["plus"]["peaks"]), len(results["minus"]["peaks"])
    adopted_side = "plus" if n_plus >= n_minus else "minus"
    n_rings = max(n_plus, n_minus)
    print(f"[STEP4] + 側 {n_plus} 本 / - 側 {n_minus} 本 → 多い方【{adopted_side}】を採用")
    print(f"[結果]  リング数: {n_rings} 本")
    ad = results[adopted_side]
    print("        採用側リング半径[px]: "
          f"{[(round(float(r), 1), rn) for r, rn in zip(ad['peaks'], ad['regions'])]}")

    # --- 各側の図・CSV を保存 ---
    for side in ("plus", "minus"):
        r = results[side]
        save_side_outputs(side, r["t"], r["values"], r["peaks"], r["regions"],
                          p, out_prefix, adopted=(side == adopted_side))

    # --- アノテーション画像（両側を描画、採用側を強調）---
    annotated = draw_annotated(img, gray, center, angle_deg, results,
                               adopted_side, n_rings, p)
    cv2.imwrite(out_prefix + "_annotated.png", annotated)

    # --- 平滑化後の画像を保存（前処理が意図通りか目視確認するため）---
    if p.presmooth > 0:
        cv2.imwrite(out_prefix + "_smoothed.png", gray_u8)

    print(f"[出力]  {out_prefix}_annotated.png")
    if p.presmooth > 0:
        print(f"[出力]  {out_prefix}_smoothed.png")
    print(f"[出力]  {out_prefix}_plus_profile.png / _minus_profile.png")
    print(f"[出力]  {out_prefix}_plus_data.csv / _minus_data.csv")


# ======================================================================
# iPhone / ブラウザ表示用エントリポイント
#   ファイルを一切書かず、表示したい 2 枚だけを PNG バイト列で返す。
#     1. annotated  : 検出位置を重ねた写真
#     2. profile    : リング数が多い方（採用側）のプロファイル図のみ
# ======================================================================
def analyze_for_display(img, p: Params, center=None, angle_deg=None,
                        axis_ratio=None, max_side=1024, dpi=110):
    """解析して (n_rings, annotated_png, profile_png, info) を返す。

    img      : BGR の numpy 配列（PIL などから作って渡す）
    max_side : 長辺をこの画素数まで縮小してから解析する。
               iPhone の写真は 4032x3024 と大きく、そのままだと遅いうえ、
               領域境界などの px 指定パラメータが合わなくなるため。
               None を渡すと縮小しない。
    """
    # --- 長辺を max_side に縮小（パラメータの互換性と速度のため）---
    scale = 1.0
    if max_side:
        h0, w0 = img.shape[:2]
        if max(h0, w0) > max_side:
            scale = max_side / max(h0, w0)
            img = cv2.resize(img, (int(round(w0 * scale)), int(round(h0 * scale))),
                             interpolation=cv2.INTER_AREA)

    gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = apply_fiji_smooth(gray_raw, p)
    gray_u8 = to_uint8(gray)

    # --- 中心・長軸角度 ---
    if center is None:
        center = find_transmitted_center(gray_u8)
    if angle_deg is None:
        angle_deg, fitted_ratio = detect_major_axis_angle(gray_u8, center)
        if angle_deg is None:
            angle_deg = 0.0
        if axis_ratio is None:
            axis_ratio = fitted_ratio

    eff_mode = p.width_mode
    if p.line_width > 1 and eff_mode == "arc" and not axis_ratio:
        eff_mode = "normal"
    ratio_arg = axis_ratio if eff_mode == "arc" else None

    # --- 両側を検出 ---
    results = {}
    for side in ("plus", "minus"):
        t, values = sample_halfline_profile(gray, center, angle_deg, side, p,
                                            axis_ratio=ratio_arg)
        peaks, peak_regions = detect_hybrid_oneside(t, values, p)
        results[side] = {"t": t, "values": values,
                         "peaks": peaks, "regions": peak_regions}

    n_plus, n_minus = len(results["plus"]["peaks"]), len(results["minus"]["peaks"])
    adopted_side = "plus" if n_plus >= n_minus else "minus"
    n_rings = max(n_plus, n_minus)

    # --- 採用側のプロファイル図だけを生成（非採用側は作らない）---
    ad = results[adopted_side]
    profile_png = save_side_outputs(
        adopted_side, ad["t"], ad["values"], ad["peaks"], ad["regions"],
        p, out_prefix=None, adopted=True, return_png=True, dpi=dpi)

    # --- アノテーション画像（両側を描画、採用側を強調）---
    annotated = draw_annotated(img, gray, center, angle_deg, results,
                               adopted_side, n_rings, p)
    ok, enc = cv2.imencode(".png", annotated)
    annotated_png = enc.tobytes() if ok else None

    info = {
        "n_rings": n_rings,
        "adopted_side": adopted_side,
        "n_plus": n_plus,
        "n_minus": n_minus,
        "center": center,
        "angle_deg": angle_deg,
        "axis_ratio": axis_ratio,
        "width_mode": eff_mode,
        "scale": scale,
        "radii": [(round(float(r), 1), rn)
                  for r, rn in zip(ad["peaks"], ad["regions"])],
    }
    return n_rings, annotated_png, profile_png, info


if __name__ == "__main__":
    main()
