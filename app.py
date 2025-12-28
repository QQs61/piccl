"""
图像信号处理分析系统 - Flask后端
功能：图像分割、增强与复原、边缘检测、消噪、压缩等处理及可视化分析
学术风格版本 V2.1
"""

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from PIL import Image
import os
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import json

# ==================== 解决中文字体问题 ====================
def setup_font():
    """设置字体，优先使用系统中文字体"""
    import platform
    system = platform.system()
    
    if system == 'Windows':
        font_list = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
    elif system == 'Darwin':
        font_list = ['Arial Unicode MS', 'Heiti SC', 'STHeiti']
    else:
        font_list = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
    
    from matplotlib.font_manager import fontManager
    available = set([f.name for f in fontManager.ttflist])
    
    for font in font_list:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return True
    
    # 如果没有中文字体，使用默认英文
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return False

HAS_CHINESE_FONT = setup_font()

app = Flask(__name__)
app.secret_key = 'image_processing_system_2024'

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

# 服务器端存储处理结果（解决localStorage超限问题）
RESULTS_CACHE = {}


def image_to_base64(img, is_gray=False):
    """将OpenCV图像转换为base64"""
    if is_gray:
        img_pil = Image.fromarray(img)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
    buffer = BytesIO()
    img_pil.save(buffer, format='PNG')
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"


def plot_to_base64(fig):
    """将matplotlib图表转换为base64"""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='#ffffff')
    buffer.seek(0)
    result = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
    plt.close(fig)
    return result


def calculate_metrics(original, processed):
    """计算图像质量指标"""
    # 转换为灰度进行比较
    if len(original.shape) == 3:
        g1 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        g1 = original
    
    if len(processed.shape) == 3:
        g2 = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        g2 = processed
    
    # 如果尺寸不同，调整为相同尺寸
    if g1.shape != g2.shape:
        h = min(g1.shape[0], g2.shape[0])
        w = min(g1.shape[1], g2.shape[1])
        g1 = g1[:h, :w]
        g2 = g2[:h, :w]
    
    mse = float(np.mean((g1.astype(float) - g2.astype(float)) ** 2))
    psnr = float(10 * np.log10((255 ** 2) / mse)) if mse > 0 else float('inf')
    
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1, mu2 = float(g1.mean()), float(g2.mean())
    s1, s2 = float(g1.var()), float(g2.var())
    s12 = float(np.cov(g1.flatten(), g2.flatten())[0, 1])
    ssim = ((2*mu1*mu2+c1)*(2*s12+c2)) / ((mu1**2+mu2**2+c1)*(s1+s2+c2))
    
    return {'mse': round(mse, 4), 'psnr': round(psnr, 4) if psnr != float('inf') else 'INF', 'ssim': round(float(ssim), 4)}


def create_histogram_chart(original, processed):
    """创建直方图分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Histogram Analysis', fontsize=14, fontweight='bold')
    
    # 处理可能不同尺寸的图像
    orig_display = original
    proc_display = processed
    
    axes[0, 0].set_title('Original Image')
    if len(original.shape) == 3:
        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].axis('off')
    
    axes[0, 1].set_title('Processed Image')
    if len(processed.shape) == 3:
        axes[0, 1].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    else:
        axes[0, 1].imshow(processed, cmap='gray')
    axes[0, 1].axis('off')
    
    # 计算直方图（转为相同格式）
    if len(original.shape) == 3:
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        orig_gray = original
    
    if len(processed.shape) == 3:
        proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        proc_gray = processed
    
    axes[1, 0].set_title('Original Histogram')
    hist = cv2.calcHist([orig_gray], [0], None, [256], [0, 256])
    axes[1, 0].fill_between(range(256), hist.flatten(), alpha=0.7, color='steelblue')
    axes[1, 0].set_xlim([0, 256])
    axes[1, 0].set_xlabel('Pixel Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Processed Histogram')
    hist = cv2.calcHist([proc_gray], [0], None, [256], [0, 256])
    axes[1, 1].fill_between(range(256), hist.flatten(), alpha=0.7, color='coral')
    axes[1, 1].set_xlim([0, 256])
    axes[1, 1].set_xlabel('Pixel Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plot_to_base64(fig)


def create_frequency_chart(original, processed):
    """创建频域分析图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Frequency Domain Analysis', fontsize=14, fontweight='bold')
    
    g1 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if len(original.shape) == 3 else original
    g2 = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY) if len(processed.shape) == 3 else processed
    
    axes[0, 0].imshow(g1, cmap='gray')
    axes[0, 0].set_title('Original (Gray)')
    axes[0, 0].axis('off')
    
    f1 = np.fft.fftshift(np.fft.fft2(g1))
    mag1 = 20 * np.log(np.abs(f1) + 1)
    im1 = axes[0, 1].imshow(mag1, cmap='jet')
    axes[0, 1].set_title('Original Spectrum')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    phase1 = np.angle(f1)
    im2 = axes[0, 2].imshow(phase1, cmap='hsv')
    axes[0, 2].set_title('Original Phase')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    axes[1, 0].imshow(g2, cmap='gray')
    axes[1, 0].set_title('Processed (Gray)')
    axes[1, 0].axis('off')
    
    f2 = np.fft.fftshift(np.fft.fft2(g2))
    mag2 = 20 * np.log(np.abs(f2) + 1)
    im3 = axes[1, 1].imshow(mag2, cmap='jet')
    axes[1, 1].set_title('Processed Spectrum')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)
    
    phase2 = np.angle(f2)
    im4 = axes[1, 2].imshow(phase2, cmap='hsv')
    axes[1, 2].set_title('Processed Phase')
    axes[1, 2].axis('off')
    plt.colorbar(im4, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    return plot_to_base64(fig)


# ==================== 图像分割 ====================

def threshold_segmentation(img, params=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Fixed Threshold Segmentation Analysis', fontsize=14, fontweight='bold')
    
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Grayscale')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(binary, cmap='gray')
    axes[0, 2].set_title('Binary (T=127)')
    axes[0, 2].axis('off')
    
    for i, t in enumerate([50, 100, 150]):
        _, r = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
        axes[1, i].imshow(r, cmap='gray')
        axes[1, i].set_title(f'Threshold = {t}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    chart = plot_to_base64(fig)
    
    return binary, True, chart, {'method': 'Fixed Threshold', 'threshold': 127,
        'foreground_ratio': round(float(np.sum(binary==255))/binary.size*100, 2)}


def otsu_segmentation(img, params=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    thresh_val, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Otsu Segmentation (Optimal T={int(thresh_val)})', fontsize=14, fontweight='bold')
    
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(binary, cmap='gray')
    axes[0, 1].set_title('Otsu Result')
    axes[0, 1].axis('off')
    
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    axes[0, 2].fill_between(range(256), hist.flatten(), alpha=0.7, color='steelblue')
    axes[0, 2].axvline(x=thresh_val, color='red', linestyle='--', linewidth=2, label=f'T={int(thresh_val)}')
    axes[0, 2].set_title('Histogram + Threshold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Between-class variance
    variances = []
    for t in range(0, 256, 2):
        w0 = np.sum(gray <= t) / gray.size
        w1 = 1 - w0
        if w0 == 0 or w1 == 0:
            variances.append(0)
            continue
        u0 = np.mean(gray[gray <= t]) if np.any(gray <= t) else 0
        u1 = np.mean(gray[gray > t]) if np.any(gray > t) else 0
        variances.append(w0 * w1 * (u0 - u1) ** 2)
    
    axes[1, 0].plot(range(0, 256, 2), variances, 'b-', linewidth=1.5)
    axes[1, 0].axvline(x=thresh_val, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_title('Between-class Variance')
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].grid(True, alpha=0.3)
    
    sizes = [np.sum(binary == 255), np.sum(binary == 0)]
    axes[1, 1].pie(sizes, labels=['Foreground', 'Background'], colors=['#ff9999', '#66b3ff'], autopct='%1.1f%%')
    axes[1, 1].set_title('Area Ratio')
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    axes[1, 2].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(f'Contours: {len(contours)}')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    chart = plot_to_base64(fig)
    
    return binary, True, chart, {'method': 'Otsu Threshold', 'optimal_threshold': int(thresh_val),
        'contours': int(len(contours))}


def kmeans_segmentation(img, params=None):
    k = 3
    pixels = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    result = centers[labels.flatten()].reshape(img.shape)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'K-means Clustering (K={k})', fontsize=14, fontweight='bold')
    
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'K-means (K={k})')
    axes[0, 1].axis('off')
    
    axes[0, 2].bar(range(k), [1]*k, color=centers[:, ::-1]/255.0, edgecolor='black')
    axes[0, 2].set_title('Cluster Centers')
    axes[0, 2].set_xticks(range(k))
    
    for i, tk in enumerate([2, 4, 6]):
        _, tl, tc = cv2.kmeans(pixels, tk, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        tc = np.uint8(tc)
        tr = tc[tl.flatten()].reshape(img.shape)
        axes[1, i].imshow(cv2.cvtColor(tr, cv2.COLOR_BGR2RGB))
        axes[1, i].set_title(f'K = {tk}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    chart = plot_to_base64(fig)
    
    return result, False, chart, {'method': 'K-means Clustering', 'k': k}


def watershed_segmentation(img, params=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.7 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    result = img.copy()
    markers = cv2.watershed(result, markers)
    result[markers == -1] = [255, 0, 0]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Watershed Segmentation', fontsize=14, fontweight='bold')
    
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(binary, cmap='gray')
    axes[0, 1].set_title('Binary')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(opening, cmap='gray')
    axes[0, 2].set_title('Opening')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(dist, cmap='jet')
    axes[0, 3].set_title('Distance Transform')
    axes[0, 3].axis('off')
    
    axes[1, 0].imshow(sure_bg, cmap='gray')
    axes[1, 0].set_title('Sure Background')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(sure_fg, cmap='gray')
    axes[1, 1].set_title('Sure Foreground')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(markers, cmap='jet')
    axes[1, 2].set_title('Markers')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[1, 3].set_title('Watershed Result')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    chart = plot_to_base64(fig)
    
    return result, False, chart, {'method': 'Watershed', 'regions': int(len(np.unique(markers)) - 1)}


# ==================== 增强与复原 ====================

def histogram_equalization(img, params=None):
    if len(img.shape) == 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        g1, g2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    else:
        result = cv2.equalizeHist(img)
        g1, g2 = img, result
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Histogram Equalization Analysis', fontsize=14, fontweight='bold')
    
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB) if len(result.shape) == 3 else result, cmap='gray')
    axes[0, 1].set_title('Equalized')
    axes[0, 1].axis('off')
    
    h1 = cv2.calcHist([g1], [0], None, [256], [0, 256])
    h2 = cv2.calcHist([g2], [0], None, [256], [0, 256])
    cdf1, cdf2 = h1.cumsum() / h1.cumsum().max(), h2.cumsum() / h2.cumsum().max()
    
    axes[0, 2].plot(cdf1, 'b-', linewidth=2, label='Original')
    axes[0, 2].plot(cdf2, 'r-', linewidth=2, label='Equalized')
    axes[0, 2].set_title('CDF Comparison')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].fill_between(range(256), h1.flatten(), alpha=0.7, color='steelblue')
    axes[1, 0].set_title('Original Histogram')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].fill_between(range(256), h2.flatten(), alpha=0.7, color='coral')
    axes[1, 1].set_title('Equalized Histogram')
    axes[1, 1].grid(True, alpha=0.3)
    
    txt = f"Original:\n  Mean={g1.mean():.1f}\n  Std={g1.std():.1f}\n\nEqualized:\n  Mean={g2.mean():.1f}\n  Std={g2.std():.1f}"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center', fontsize=11, transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat'))
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Statistics')
    
    plt.tight_layout()
    chart = plot_to_base64(fig)
    
    return result, False, chart, {'method': 'Histogram Equalization', 'original_mean': round(float(g1.mean()), 2),
        'enhanced_mean': round(float(g2.mean()), 2)}


def clahe_enhancement(img, params=None):
    clip, tile = 2.0, 8
    
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        result = clahe.apply(img)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'CLAHE Enhancement (clipLimit={clip})', fontsize=14, fontweight='bold')
    
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB) if len(result.shape) == 3 else result, cmap='gray')
    axes[0, 1].set_title('CLAHE Result')
    axes[0, 1].axis('off')
    
    if len(img.shape) == 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        he = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        he = cv2.equalizeHist(img)
    
    axes[0, 2].imshow(cv2.cvtColor(he, cv2.COLOR_BGR2RGB) if len(he.shape) == 3 else he, cmap='gray')
    axes[0, 2].set_title('Standard HE')
    axes[0, 2].axis('off')
    
    if len(result.shape) == 3:
        diff = cv2.absdiff(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), cv2.cvtColor(he, cv2.COLOR_BGR2GRAY))
    else:
        diff = cv2.absdiff(result, he)
    axes[0, 3].imshow(diff, cmap='hot')
    axes[0, 3].set_title('Difference')
    axes[0, 3].axis('off')
    
    for i, c in enumerate([1.0, 2.0, 4.0, 8.0]):
        cl = cv2.createCLAHE(clipLimit=c, tileGridSize=(tile, tile))
        if len(img.shape) == 3:
            lt = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lt[:, :, 0] = cl.apply(lt[:, :, 0])
            tr = cv2.cvtColor(lt, cv2.COLOR_LAB2BGR)
            axes[1, i].imshow(cv2.cvtColor(tr, cv2.COLOR_BGR2RGB))
        else:
            axes[1, i].imshow(cl.apply(img), cmap='gray')
        axes[1, i].set_title(f'clipLimit = {c}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    chart = plot_to_base64(fig)
    
    return result, False, chart, {'method': 'CLAHE', 'clip_limit': clip, 'tile_size': tile}


def gamma_correction(img, params=None):
    gamma = 1.5
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    result = cv2.LUT(img, table)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Gamma Correction (gamma={gamma})', fontsize=14, fontweight='bold')
    
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB) if len(result.shape) == 3 else result, cmap='gray')
    axes[0, 1].set_title(f'Gamma = {gamma}')
    axes[0, 1].axis('off')
    
    x = np.arange(256)
    for g in [0.5, 1.0, 1.5, 2.0, 2.5]:
        y = ((x / 255.0) ** (1.0 / g)) * 255
        axes[0, 2].plot(x, y, label=f'g={g}', linewidth=2)
    axes[0, 2].plot([0, 255], [0, 255], 'k--', alpha=0.5)
    axes[0, 2].set_title('Gamma Curves')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    for i, g in enumerate([0.5, 1.5, 2.5]):
        tbl = np.array([((j / 255.0) ** (1.0 / g)) * 255 for j in range(256)]).astype("uint8")
        tr = cv2.LUT(img, tbl)
        axes[1, i].imshow(cv2.cvtColor(tr, cv2.COLOR_BGR2RGB) if len(tr.shape) == 3 else tr, cmap='gray')
        effect = '(Brighten)' if g < 1 else '(Darken)'
        axes[1, i].set_title(f'g = {g} {effect}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    chart = plot_to_base64(fig)
    
    return result, False, chart, {'method': 'Gamma Correction', 'gamma': gamma}


def unsharp_masking(img, params=None):
    sigma, strength = 1.0, 1.5
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    result = cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Unsharp Masking (strength={strength})', fontsize=14, fontweight='bold')
    
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB) if len(blurred.shape) == 3 else blurred, cmap='gray')
    axes[0, 1].set_title('Gaussian Blur')
    axes[0, 1].axis('off')
    
    mask = cv2.absdiff(img, blurred)
    axes[0, 2].imshow(mask * 3, cmap='gray')
    axes[0, 2].set_title('Mask (3x)')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB) if len(result.shape) == 3 else result, cmap='gray')
    axes[1, 0].set_title('Sharpened')
    axes[1, 0].axis('off')
    
    for i, s in enumerate([0.5, 2.0]):
        tr = cv2.addWeighted(img, 1 + s, blurred, -s, 0)
        axes[1, i + 1].imshow(cv2.cvtColor(tr, cv2.COLOR_BGR2RGB) if len(tr.shape) == 3 else tr, cmap='gray')
        axes[1, i + 1].set_title(f'Strength = {s}')
        axes[1, i + 1].axis('off')
    
    plt.tight_layout()
    chart = plot_to_base64(fig)
    
    return result, False, chart, {'method': 'Unsharp Masking', 'strength': strength}


def wiener_filter(img, params=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    ks = 15
    kernel = np.zeros((ks, ks))
    kernel[ks//2, :] = 1.0 / ks
    blurred = cv2.filter2D(gray, -1, kernel)
    
    noise_var = 0.01
    kft = np.fft.fft2(kernel, s=gray.shape)
    ift = np.fft.fft2(blurred)
    wiener = np.conj(kft) / (np.abs(kft)**2 + noise_var)
    result = np.abs(np.fft.ifft2(wiener * ift))
    result = np.uint8(np.clip(result, 0, 255))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Wiener Filter Restoration', fontsize=14, fontweight='bold')
    
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(blurred, cmap='gray')
    axes[0, 1].set_title('Motion Blurred')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(result, cmap='gray')
    axes[0, 2].set_title('Wiener Restored')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(kernel, cmap='hot')
    axes[1, 0].set_title('Blur Kernel')
    axes[1, 0].axis('off')
    
    spec1 = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(blurred))) + 1)
    axes[1, 1].imshow(spec1, cmap='jet')
    axes[1, 1].set_title('Blurred Spectrum')
    axes[1, 1].axis('off')
    
    spec2 = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(result))) + 1)
    axes[1, 2].imshow(spec2, cmap='jet')
    axes[1, 2].set_title('Restored Spectrum')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    chart = plot_to_base64(fig)
    
    return result, True, chart, {'method': 'Wiener Filter', 'kernel_size': ks}


# ==================== 边缘检测 ====================

def multi_edge_detection(img, params=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    canny = cv2.Canny(gray, 50, 150)
    sx, sy = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.uint8(np.clip(np.sqrt(sx**2 + sy**2), 0, 255))
    laplacian = np.uint8(np.absolute(cv2.Laplacian(gray, cv2.CV_64F)))
    
    kx, ky = np.array([[1,1,1],[0,0,0],[-1,-1,-1]]), np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    px, py = cv2.filter2D(gray, cv2.CV_64F, kx), cv2.filter2D(gray, cv2.CV_64F, ky)
    prewitt = np.uint8(np.clip(np.sqrt(px**2 + py**2), 0, 255))
    
    rx, ry = np.array([[1,0],[0,-1]]), np.array([[0,1],[-1,0]])
    roberts = np.uint8(np.clip(np.sqrt(cv2.filter2D(gray, cv2.CV_64F, rx)**2 + 
                                        cv2.filter2D(gray, cv2.CV_64F, ry)**2), 0, 255))
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Multi-Operator Edge Detection', fontsize=14, fontweight='bold')
    
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Grayscale')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(canny, cmap='gray')
    axes[0, 1].set_title('Canny')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(sobel, cmap='gray')
    axes[0, 2].set_title('Sobel')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(laplacian, cmap='gray')
    axes[1, 0].set_title('Laplacian')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(prewitt, cmap='gray')
    axes[1, 1].set_title('Prewitt')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(roberts, cmap='gray')
    axes[1, 2].set_title('Roberts')
    axes[1, 2].axis('off')
    
    direction = np.arctan2(sy, sx) * 180 / np.pi
    im = axes[2, 0].imshow(direction, cmap='hsv')
    axes[2, 0].set_title('Gradient Direction')
    axes[2, 0].axis('off')
    plt.colorbar(im, ax=axes[2, 0], fraction=0.046)
    
    methods = ['Canny', 'Sobel', 'Laplacian', 'Prewitt', 'Roberts']
    counts = [np.sum(canny > 0), np.sum(sobel > 50), np.sum(laplacian > 50), np.sum(prewitt > 50), np.sum(roberts > 30)]
    axes[2, 1].bar(methods, counts, color=['steelblue', 'coral', 'green', 'purple', 'orange'])
    axes[2, 1].set_title('Edge Pixel Count')
    axes[2, 1].tick_params(axis='x', rotation=45)
    
    sk = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    axes[2, 2].imshow(sk, cmap='RdBu', vmin=-2, vmax=2)
    axes[2, 2].set_title('Sobel X Kernel')
    for i in range(3):
        for j in range(3):
            axes[2, 2].text(j, i, sk[i, j], ha='center', va='center', fontsize=12)
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    chart = plot_to_base64(fig)
    
    return canny, True, chart, {'method': 'Multi-Operator Edge Detection', 'canny_edges': int(counts[0])}


def canny_analysis(img, params=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    
    sx, sy = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sx**2 + sy**2)
    direction = np.arctan2(sy, sx) * 180 / np.pi
    
    canny = cv2.Canny(gray, 50, 150)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Canny Edge Detection Analysis', fontsize=14, fontweight='bold')
    
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('1. Grayscale')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(blurred, cmap='gray')
    axes[0, 1].set_title('2. Gaussian Blur')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(magnitude, cmap='gray')
    axes[0, 2].set_title('3. Gradient Magnitude')
    axes[0, 2].axis('off')
    
    im = axes[0, 3].imshow(direction, cmap='hsv')
    axes[0, 3].set_title('4. Gradient Direction')
    axes[0, 3].axis('off')
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046)
    
    axes[1, 0].imshow(canny, cmap='gray')
    axes[1, 0].set_title('5. Canny Result')
    axes[1, 0].axis('off')
    
    for i, (l, h) in enumerate([(30, 90), (50, 150), (100, 200)]):
        tc = cv2.Canny(gray, l, h)
        axes[1, i + 1].imshow(tc, cmap='gray')
        axes[1, i + 1].set_title(f'T=({l}, {h})')
        axes[1, i + 1].axis('off')
    
    plt.tight_layout()
    chart = plot_to_base64(fig)
    
    return canny, True, chart, {'method': 'Canny Edge Detection', 'edge_pixels': int(np.sum(canny > 0)),
        'edge_ratio': round(float(np.sum(canny > 0)) / canny.size * 100, 2)}


# ==================== 消噪 ====================

def multi_denoise(img, params=None):
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    noisy = cv2.add(img, noise)
    
    gaussian = cv2.GaussianBlur(noisy, (5, 5), 0)
    median = cv2.medianBlur(noisy, 5)
    bilateral = cv2.bilateralFilter(noisy, 9, 75, 75)
    nlm = cv2.fastNlMeansDenoisingColored(noisy, None, 10, 10, 7, 21) if len(noisy.shape) == 3 else cv2.fastNlMeansDenoising(noisy, None, 10, 7, 21)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Denoising Methods Comparison', fontsize=14, fontweight='bold')
    
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB) if len(noisy.shape) == 3 else noisy, cmap='gray')
    axes[0, 1].set_title('Noisy')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB) if len(gaussian.shape) == 3 else gaussian, cmap='gray')
    axes[0, 2].set_title('Gaussian')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB) if len(median.shape) == 3 else median, cmap='gray')
    axes[0, 3].set_title('Median')
    axes[0, 3].axis('off')
    
    axes[1, 0].imshow(cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB) if len(bilateral.shape) == 3 else bilateral, cmap='gray')
    axes[1, 0].set_title('Bilateral')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(nlm, cv2.COLOR_BGR2RGB) if len(nlm.shape) == 3 else nlm, cmap='gray')
    axes[1, 1].set_title('NLM')
    axes[1, 1].axis('off')
    
    methods = ['Gaussian', 'Median', 'Bilateral', 'NLM']
    psnrs = []
    for d in [gaussian, median, bilateral, nlm]:
        mse = np.mean((img.astype(float) - d.astype(float)) ** 2)
        psnrs.append(10 * np.log10((255 ** 2) / mse) if mse > 0 else 100)
    
    axes[1, 2].bar(methods, psnrs, color=['steelblue', 'coral', 'green', 'purple'])
    axes[1, 2].set_title('PSNR Comparison')
    axes[1, 2].set_ylabel('PSNR (dB)')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    best = methods[np.argmax(psnrs)]
    txt = f'Best Method:\n{best}\n\nPSNR:\n{max(psnrs):.1f} dB'
    axes[1, 3].text(0.5, 0.5, txt, ha='center', va='center', fontsize=12, transform=axes[1, 3].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightyellow'))
    axes[1, 3].axis('off')
    axes[1, 3].set_title('Summary')
    
    plt.tight_layout()
    chart = plot_to_base64(fig)
    
    return nlm, False, chart, {'method': 'Multi-Method Denoising', 'best': best, 'best_psnr': round(float(max(psnrs)), 2)}


def gaussian_denoise(img, params=None):
    ks = 5
    result = cv2.GaussianBlur(img, (ks, ks), 0)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Gaussian Filter (kernel={ks}x{ks})', fontsize=14, fontweight='bold')
    
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB) if len(result.shape) == 3 else result, cmap='gray')
    axes[0, 1].set_title('Filtered')
    axes[0, 1].axis('off')
    
    gk = cv2.getGaussianKernel(ks, 0)
    gk2d = gk @ gk.T
    im = axes[0, 2].imshow(gk2d, cmap='hot')
    axes[0, 2].set_title('Gaussian Kernel')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
    
    # 显示核的剖面曲线代替3D图
    axes[0, 3].plot(gk2d[ks//2, :], 'b-', linewidth=2, label='Horizontal')
    axes[0, 3].plot(gk2d[:, ks//2], 'r--', linewidth=2, label='Vertical')
    axes[0, 3].set_title('Kernel Profile')
    axes[0, 3].legend()
    axes[0, 3].grid(True, alpha=0.3)
    
    for i, k in enumerate([3, 7, 11, 15]):
        tr = cv2.GaussianBlur(img, (k, k), 0)
        axes[1, i].imshow(cv2.cvtColor(tr, cv2.COLOR_BGR2RGB) if len(tr.shape) == 3 else tr, cmap='gray')
        axes[1, i].set_title(f'Kernel = {k}x{k}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    chart = plot_to_base64(fig)
    
    return result, False, chart, {'method': 'Gaussian Filter', 'kernel_size': ks}


def median_denoise(img, params=None):
    ks = 5
    result = cv2.medianBlur(img, ks)
    
    # Add salt & pepper noise demo
    noisy = img.copy()
    salt, pepper = np.random.random(img.shape[:2]) < 0.02, np.random.random(img.shape[:2]) < 0.02
    if len(img.shape) == 3:
        noisy[salt], noisy[pepper] = [255, 255, 255], [0, 0, 0]
    else:
        noisy[salt], noisy[pepper] = 255, 0
    
    noisy_med = cv2.medianBlur(noisy, ks)
    noisy_gauss = cv2.GaussianBlur(noisy, (ks, ks), 0)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Median Filter (kernel={ks}x{ks})', fontsize=14, fontweight='bold')
    
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB) if len(noisy.shape) == 3 else noisy, cmap='gray')
    axes[0, 1].set_title('Salt & Pepper')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cv2.cvtColor(noisy_med, cv2.COLOR_BGR2RGB) if len(noisy_med.shape) == 3 else noisy_med, cmap='gray')
    axes[0, 2].set_title('Median Filtered')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(cv2.cvtColor(noisy_gauss, cv2.COLOR_BGR2RGB) if len(noisy_gauss.shape) == 3 else noisy_gauss, cmap='gray')
    axes[1, 0].set_title('Gaussian Filtered')
    axes[1, 0].axis('off')
    
    mse1 = np.mean((img.astype(float) - noisy_med.astype(float)) ** 2)
    mse2 = np.mean((img.astype(float) - noisy_gauss.astype(float)) ** 2)
    p1 = 10 * np.log10((255 ** 2) / mse1) if mse1 > 0 else 100
    p2 = 10 * np.log10((255 ** 2) / mse2) if mse2 > 0 else 100
    
    axes[1, 1].bar(['Median', 'Gaussian'], [p1, p2], color=['coral', 'steelblue'])
    axes[1, 1].set_title('PSNR Comparison')
    axes[1, 1].set_ylabel('PSNR (dB)')
    
    txt = 'Median Filter:\n- Non-linear\n- Best for salt & pepper\n- Preserves edges'
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center', fontsize=11, transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightyellow'))
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Properties')
    
    plt.tight_layout()
    chart = plot_to_base64(fig)
    
    return result, False, chart, {'method': 'Median Filter', 'kernel_size': ks}


# ==================== 压缩 ====================

def jpeg_compression(img, params=None):
    qualities = [10, 30, 50, 70, 90]
    compressed = []
    sizes = []
    psnrs = []
    
    for q in qualities:
        _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR if len(img.shape) == 3 else cv2.IMREAD_GRAYSCALE)
        compressed.append(dec)
        sizes.append(len(enc))
        mse = np.mean((img.astype(float) - dec.astype(float)) ** 2)
        psnrs.append(10 * np.log10((255 ** 2) / mse) if mse > 0 else 100)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('JPEG Compression Analysis', fontsize=14, fontweight='bold')
    
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for pos, q, c in zip(positions, qualities, compressed):
        axes[pos].imshow(cv2.cvtColor(c, cv2.COLOR_BGR2RGB) if len(c.shape) == 3 else c, cmap='gray')
        axes[pos].set_title(f'Q = {q}')
        axes[pos].axis('off')
    
    orig_size = img.nbytes
    ratios = [orig_size / s for s in sizes]
    axes[2, 0].plot(qualities, ratios, 'bo-', linewidth=2, markersize=8)
    axes[2, 0].set_title('Compression Ratio')
    axes[2, 0].set_xlabel('Quality')
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].plot(qualities, psnrs, 'ro-', linewidth=2, markersize=8)
    axes[2, 1].set_title('PSNR')
    axes[2, 1].set_xlabel('Quality')
    axes[2, 1].set_ylabel('dB')
    axes[2, 1].grid(True, alpha=0.3)
    
    axes[2, 2].bar(qualities, [s / 1024 for s in sizes], color='steelblue')
    axes[2, 2].set_title('File Size')
    axes[2, 2].set_xlabel('Quality')
    axes[2, 2].set_ylabel('KB')
    
    plt.tight_layout()
    chart = plot_to_base64(fig)
    
    return compressed[2], False, chart, {'method': 'JPEG Compression', 'original_kb': round(float(orig_size) / 1024, 2)}


def dct_compression(img, params=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape
    h_new = (h // 8) * 8
    w_new = (w // 8) * 8
    gray = gray[:h_new, :w_new].astype(np.float32)
    
    dct = np.zeros_like(gray)
    for i in range(0, h_new, 8):
        for j in range(0, w_new, 8):
            dct[i:i+8, j:j+8] = cv2.dct(gray[i:i+8, j:j+8])
    
    ratios = [0.1, 0.3, 0.5, 0.7]
    recons = []
    
    for r in ratios:
        q = dct.copy()
        for i in range(0, h_new, 8):
            for j in range(0, w_new, 8):
                mask = np.zeros((8, 8))
                keep = int(64 * r)
                for k in range(keep):
                    mask[min(k // 8, 7), min(k % 8, 7)] = 1
                q[i:i+8, j:j+8] *= mask
        
        rec = np.zeros_like(gray)
        for i in range(0, h_new, 8):
            for j in range(0, w_new, 8):
                rec[i:i+8, j:j+8] = cv2.idct(q[i:i+8, j:j+8])
        recons.append(np.clip(rec, 0, 255).astype(np.uint8))
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('DCT Compression Analysis', fontsize=14, fontweight='bold')
    
    axes[0, 0].imshow(gray.astype(np.uint8), cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    dct_vis = np.log(np.abs(dct) + 1)
    axes[0, 1].imshow(dct_vis, cmap='jet')
    axes[0, 1].set_title('DCT Coefficients')
    axes[0, 1].axis('off')
    
    basis = np.zeros((64, 64))
    for i in range(8):
        for j in range(8):
            b = np.zeros((8, 8))
            b[i, j] = 1
            basis[i*8:(i+1)*8, j*8:(j+1)*8] = cv2.idct(b)
    axes[0, 2].imshow(basis, cmap='gray')
    axes[0, 2].set_title('DCT Basis')
    axes[0, 2].axis('off')
    
    zigzag = np.array([[0,1,5,6,14,15,27,28],[2,4,7,13,16,26,29,42],[3,8,12,17,25,30,41,43],
                       [9,11,18,24,31,40,44,53],[10,19,23,32,39,45,52,54],[20,22,33,38,46,51,55,60],
                       [21,34,37,47,50,56,59,61],[35,36,48,49,57,58,62,63]])
    axes[0, 3].imshow(zigzag, cmap='viridis')
    axes[0, 3].set_title('Zigzag Order')
    for i in range(8):
        for j in range(8):
            axes[0, 3].text(j, i, str(zigzag[i, j]), ha='center', va='center', fontsize=7, color='white')
    axes[0, 3].axis('off')
    
    for i, (r, rec) in enumerate(zip(ratios, recons)):
        mse = float(np.mean((gray - rec.astype(float)) ** 2))
        psnr = 10 * np.log10((255 ** 2) / mse) if mse > 0 else 100
        axes[1, i].imshow(rec, cmap='gray')
        axes[1, i].set_title(f'Keep {int(r*100)}%\nPSNR={psnr:.1f}dB')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    chart = plot_to_base64(fig)
    
    return recons[2], True, chart, {'method': 'DCT Compression', 'block_size': '8x8', 
                                     'original_size': f'{h_new}x{w_new}'}


# ==================== 处理函数映射 ====================

PROCESS_FUNCTIONS = {
    'threshold': threshold_segmentation,
    'otsu': otsu_segmentation,
    'kmeans': kmeans_segmentation,
    'watershed': watershed_segmentation,
    'histogram_eq': histogram_equalization,
    'clahe': clahe_enhancement,
    'gamma': gamma_correction,
    'sharpen': unsharp_masking,
    'wiener': wiener_filter,
    'edge_multi': multi_edge_detection,
    'canny': canny_analysis,
    'denoise_multi': multi_denoise,
    'gaussian_denoise': gaussian_denoise,
    'median_denoise': median_denoise,
    'jpeg': jpeg_compression,
    'dct': dct_compression,
}

PROCESS_NAMES = {
    'threshold': 'Fixed Threshold',
    'otsu': 'Otsu Threshold',
    'kmeans': 'K-means Clustering',
    'watershed': 'Watershed',
    'histogram_eq': 'Histogram Equalization',
    'clahe': 'CLAHE',
    'gamma': 'Gamma Correction',
    'sharpen': 'Unsharp Masking',
    'wiener': 'Wiener Filter',
    'edge_multi': 'Multi-Operator',
    'canny': 'Canny Detection',
    'denoise_multi': 'Multi-Method',
    'gaussian_denoise': 'Gaussian Filter',
    'median_denoise': 'Median Filter',
    'jpeg': 'JPEG Analysis',
    'dct': 'DCT Compression',
}


# ==================== 路由 ====================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result')
def result():
    return render_template('result.html')


@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file'}), 400
    
    process_type = request.form.get('process_type', 'otsu')
    
    try:
        print(f"[INFO] Processing: {process_type}")
        
        file_bytes = file.read()
        print(f"[INFO] File size: {len(file_bytes)} bytes")
        
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Cannot read image - unsupported format'}), 400
        
        print(f"[INFO] Image shape: {img.shape}")
        
        h, w = img.shape[:2]
        if max(h, w) > 800:
            scale = 800 / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale)
            print(f"[INFO] Resized to: {img.shape}")
        
        start = time.time()
        
        if process_type not in PROCESS_FUNCTIONS:
            return jsonify({'error': f'Unknown type: {process_type}'}), 400
        
        print(f"[INFO] Running {process_type}...")
        result, is_gray, analysis_chart, stats = PROCESS_FUNCTIONS[process_type](img)
        print(f"[INFO] Algorithm done, result shape: {result.shape}")
        
        process_time = round(time.time() - start, 3)
        
        print("[INFO] Calculating metrics...")
        metrics = calculate_metrics(img, result)
        
        print("[INFO] Creating histogram chart...")
        histogram_chart = create_histogram_chart(img, result)
        
        print("[INFO] Creating frequency chart...")
        frequency_chart = create_frequency_chart(img, result)
        
        print("[INFO] Converting to base64...")
        original_b64 = image_to_base64(img)
        result_b64 = image_to_base64(result, is_gray)
        
        # 生成唯一ID并存储到服务器（解决localStorage超限问题）
        import uuid
        result_id = str(uuid.uuid4())[:8]
        
        RESULTS_CACHE[result_id] = {
            'success': True,
            'original': original_b64,
            'result': result_b64,
            'analysis_chart': analysis_chart,
            'histogram_chart': histogram_chart,
            'frequency_chart': frequency_chart,
            'metrics': metrics,
            'stats': stats,
            'process_time': process_time,
            'process_name': PROCESS_NAMES.get(process_type, process_type)
        }
        
        # 清理旧缓存（保留最近10个）
        if len(RESULTS_CACHE) > 10:
            oldest = list(RESULTS_CACHE.keys())[0]
            del RESULTS_CACHE[oldest]
        
        print(f"[INFO] Done in {process_time}s, result_id: {result_id}")
        
        return jsonify({
            'success': True,
            'result_id': result_id,
            'process_name': PROCESS_NAMES.get(process_type, process_type)
        })
            
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"[ERROR] {error_msg}")
        return jsonify({'error': str(e), 'detail': error_msg}), 500


@app.route('/get_result/<result_id>')
def get_result(result_id):
    """获取处理结果数据"""
    if result_id in RESULTS_CACHE:
        return jsonify(RESULTS_CACHE[result_id])
    return jsonify({'error': 'Result not found'}), 404


@app.route('/process_batch', methods=['POST'])
def process_batch():
    """批量处理多个算法"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file'}), 400
    
    algorithms_json = request.form.get('algorithms', '[]')
    try:
        algorithms = json.loads(algorithms_json)
    except:
        return jsonify({'error': 'Invalid algorithms'}), 400
    
    if not algorithms:
        return jsonify({'error': 'No algorithms selected'}), 400
    
    try:
        print(f"[INFO] Batch processing: {algorithms}")
        
        file_bytes = file.read()
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Cannot read image'}), 400
        
        h, w = img.shape[:2]
        if max(h, w) > 800:
            scale = 800 / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale)
        
        original_b64 = image_to_base64(img)
        
        results = []
        total_start = time.time()
        
        for algo in algorithms:
            if algo not in PROCESS_FUNCTIONS:
                continue
            
            print(f"[INFO] Processing {algo}...")
            start = time.time()
            
            result_img, is_gray, analysis_chart, stats = PROCESS_FUNCTIONS[algo](img)
            process_time = round(time.time() - start, 3)
            
            metrics = calculate_metrics(img, result_img)
            histogram_chart = create_histogram_chart(img, result_img)
            frequency_chart = create_frequency_chart(img, result_img)
            result_b64 = image_to_base64(result_img, is_gray)
            
            results.append({
                'algorithm': algo,
                'original': original_b64,
                'result': result_b64,
                'analysis_chart': analysis_chart,
                'histogram_chart': histogram_chart,
                'frequency_chart': frequency_chart,
                'metrics': metrics,
                'stats': stats,
                'process_time': process_time
            })
            
            print(f"[INFO] {algo} done in {process_time}s")
        
        total_time = round(time.time() - total_start, 2)
        
        import uuid
        result_id = str(uuid.uuid4())[:8]
        
        RESULTS_CACHE[result_id] = {
            'results': results,
            'total_time': total_time
        }
        
        if len(RESULTS_CACHE) > 10:
            oldest = list(RESULTS_CACHE.keys())[0]
            del RESULTS_CACHE[oldest]
        
        print(f"[INFO] Batch done, {len(results)} results, total {total_time}s")
        
        return jsonify({
            'success': True,
            'result_id': result_id,
            'count': len(results)
        })
        
    except Exception as e:
        import traceback
        print(f"[ERROR] {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("  Digital Image Processing Analysis System")
    print("  Visit: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
