import h5py
import matplotlib.pyplot as plt
import os

# 設定路徑
HDF5_PATH = r"C:\Users\Michael\evlaformer_lab\data\output\dataset_v1.hdf5"
SAVE_PATH = r"C:\Users\Michael\evlaformer_lab\docs\images\visual_validation.png"

# 確保 docs/images 資料夾存在
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

def visualize_standard_config():
    with h5py.File(HDF5_PATH, 'r') as f:
        # 讀取第 0 幀數據
        rgb = f['rgb'][0]
        semantic = f['semantic'][0]
        
        # 建立並排圖表 (1 row, 2 columns)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
        
        # 左圖：RGB 原圖
        axes[0].imshow(rgb)
        axes[0].set_title("Standardized Input: RGB", fontsize=14, fontweight='bold')
        axes[0].axis('off') # 隱藏座標軸
        
        # 右圖：彩色編碼遮罩 (使用 'viridis' 或 'cityscapes' 風格調色盤)
        im = axes[1].imshow(semantic, cmap='viridis')
        axes[1].set_title("Ground Truth: Semantic Mask", fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # 調整布局
        plt.tight_layout()
        
        # 存檔至 docs/images 以供 README 使用
        plt.savefig(SAVE_PATH, bbox_inches='tight')
        print(f"✅ 標準化影像已生成並存檔至: {SAVE_PATH}")
        
        # 顯示視窗
        plt.show()

if __name__ == "__main__":
    if os.path.exists(HDF5_PATH):
        visualize_standard_config()
    else:
        print(f"❌ 找不到 HDF5 檔案，請先執行 generate_data.py")