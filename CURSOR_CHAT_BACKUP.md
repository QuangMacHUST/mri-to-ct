# 💬 Cursor Chat Backup & Restore Guide

Hướng dẫn backup và khôi phục cuộc trò chuyện trong Cursor IDE khi đổi máy tính.

## 📍 **Vị trí lưu trữ Chat History**

### **Windows:**
```
C:\Users\<username>\AppData\Roaming\Cursor\User\workspaceStorage\<workspace-hash>\
```

### **macOS:**
```
~/Library/Application Support/Cursor/User/workspaceStorage/<workspace-hash>/
```

### **Linux:**
```
~/.config/Cursor/User/workspaceStorage/<workspace-hash>/
```

## 💾 **Phương pháp Backup**

### **Method 1: Manual Backup (Recommended)**
1. **Tìm workspace hash:**
   - Mở Cursor → Help → Developer Tools → Console
   - Chạy: `console.log(window.workspaceId)`
   
2. **Backup folder:**
   ```bash
   # Windows
   xcopy "C:\Users\<username>\AppData\Roaming\Cursor\User\workspaceStorage\<workspace-hash>" "D:\backup\cursor-chat\" /E /I
   
   # macOS/Linux  
   cp -r "~/Library/Application Support/Cursor/User/workspaceStorage/<workspace-hash>" ~/backup/cursor-chat/
   ```

### **Method 2: Export Chat (Manual)**
1. **Copy cuộc trò chuyện:**
   - Mở chat tab
   - Ctrl+A → Ctrl+C
   - Paste vào file `.md`

2. **Save as text file:**
   ```markdown
   # MRI-to-CT CycleGAN Project Chat History
   
   ## Ngày: [Date]
   ## Project: mri-to-ct
   
   [Paste toàn bộ cuộc trò chuyện vào đây]
   ```

## 🔄 **Restore trên máy mới**

### **Method 1: Restore từ Backup**
1. **Cài đặt Cursor** trên máy mới
2. **Clone project:**
   ```bash
   git clone <repo>
   cd mri-to-ct
   ```
3. **Mở project trong Cursor** (tạo workspace mới)
4. **Đóng Cursor**
5. **Replace chat data:**
   ```bash
   # Tìm workspace hash mới
   # Replace folder với backup
   xcopy "D:\backup\cursor-chat\*" "C:\Users\<username>\AppData\Roaming\Cursor\User\workspaceStorage\<new-workspace-hash>\" /E /Y
   ```
6. **Restart Cursor**

### **Method 2: Manual Reference**
1. **Mở file backup `.md`**
2. **Reference commands và context** khi cần
3. **Copy-paste specific commands/code** từ backup

## ⚡ **Quick Backup Commands**

### **Pre-setup Script (Windows):**
```batch
@echo off
set CURSOR_DATA=%APPDATA%\Cursor\User\workspaceStorage
set BACKUP_DIR=D:\cursor-backup\%date:~-4,4%-%date:~-10,2%-%date:~-7,2%

echo Backing up Cursor workspaces...
xcopy "%CURSOR_DATA%" "%BACKUP_DIR%" /E /I /Y

echo Backup completed: %BACKUP_DIR%
pause
```

### **Restore Script (Windows):**
```batch
@echo off
set CURSOR_DATA=%APPDATA%\Cursor\User\workspaceStorage
set BACKUP_DIR=D:\cursor-backup\[DATE-FOLDER]

echo Warning: This will overwrite current Cursor data!
pause

xcopy "%BACKUP_DIR%" "%CURSOR_DATA%" /E /I /Y

echo Restore completed!
pause
```

## 🛡️ **Best Practices**

### **Automatic Backup:**
1. **Git repository:** Commit important conversations
2. **Cloud sync:** Backup folder lên Google Drive/OneDrive
3. **Regular export:** Export long conversations to markdown

### **Project-specific Backup:**
```bash
# Tạo folder backup trong project
mkdir project-docs/chat-history/

# Export important commands/configs
echo "# Training Commands" > project-docs/chat-history/commands.md
echo "python check_gpu.py" >> project-docs/chat-history/commands.md
echo "cd src && python train.py" >> project-docs/chat-history/commands.md
```

## 📋 **Essential Info để Remember**

Nếu không backup được chat, đây là thông tin quan trọng cần nhớ:

### **Key Commands:**
```bash
# Setup
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install nibabel SimpleITK opencv-python scikit-image matplotlib tensorboard tqdm

# Test
python check_gpu.py

# Train  
cd src && python train.py

# Monitor
tensorboard --logdir=logs
```

### **Key Config:**
- **batch_size**: 1 (cho GTX 1650 4GB)
- **n_residual_blocks**: 6 (thay vì 9)
- **num_workers**: 0 (Windows multiprocessing fix)
- **Model**: 41.2M parameters

### **Known Issues & Fixes:**
- **Negative strides**: `.copy()` trong data_loader.py
- **SSIM win_size**: Auto-adjust trong metrics.py  
- **VGG loading**: Fallback methods trong models.py

## 💡 **Tips cho lần tới**

1. **Document trong Git:**
   ```bash
   git add .
   git commit -m "Add training logs and chat summary"
   ```

2. **Create project wiki/docs:**
   - Commands used
   - Issues encountered  
   - Solutions found
   - Performance metrics

3. **Export checkpoints:**
   - Lưu best model weights
   - Config files
   - Training logs

---

**Note:** Chat history là local data, không sync qua cloud. Backup thường xuyên để không mất thông tin quan trọng! 