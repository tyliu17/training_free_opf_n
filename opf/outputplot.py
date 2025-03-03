import os
import subprocess
from datetime import datetime


def output_plot(train_loss, val_loss, lr_list, output_dir):
    # **設定輸出目錄**
    timestamp = datetime.now().strftime('%m%d%H%M')
    output_file = os.path.join(output_dir, f'{timestamp}.txt')

    # **確保目錄存在**
    os.makedirs(output_dir, exist_ok=True)

    print(len(train_loss), len(val_loss), len(lr_list))  # **確認長度**

    # **儲存數據到 txt 檔案**
    with open(output_file, 'w') as file:
        for i in range(len(train_loss)):
            if i % 5 == 0 and (i // 5) < len(val_loss):  # **確保 val_loss 沒有超出索引**
                val = val_loss[i // 5]  # **正確存入數值**
            else:
                val = "NaN"  # **改成 NaN 讓 Gnuplot 忽略**
            file.write(f"{i} {train_loss[i]} {val[1]} {lr_list[i]}\n")

    print(f"✅ 輸出內容已儲存到 {output_file}")

    # **修正 Windows 路徑格式**
    gnuplot_file = output_file.replace("\\", "/")  # **確保 Gnuplot 能識別路徑**
    loss_plot_path = os.path.join(output_dir, f'loss_plot_{timestamp}.png').replace("\\", "/")
    lr_plot_path = os.path.join(output_dir, f'lr_plot_{timestamp}.png').replace("\\", "/")

    # **定義 gnuplot 腳本**
    gnuplot_script = f"""
    set terminal pngcairo enhanced font 'Arial,12' size 800,600
    set output "{loss_plot_path}"

    set title "Training Loss & Validation Loss"
    set xlabel "Step"
    set ylabel "Value"
    set grid
    set key outside

    # **確保 Validation Loss 顯示**
    plot "{gnuplot_file}" using 1:2 with lines title "Train Loss" linecolor rgb "blue", \
        "{gnuplot_file}" using 1:3 with lines title "Val Loss" linecolor rgb "red"
    

    # **第二張圖：Learning Rate**
    set output "{lr_plot_path}"
    set title "Learning Rate"
    set xlabel "Step"
    set ylabel "Learning Rate"
    set grid
    set key outside

    # **繪製 Learning Rate**
    plot "{gnuplot_file}" using 1:4 with lines title "Learning Rate" linecolor rgb "green" lw 2

    """



    # **執行 gnuplot**
    try:
        result = subprocess.run(["gnuplot"], input=gnuplot_script, text=True, check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.stderr:
            print("⚠️ Gnuplot Error:", result.stderr)
        else:
            print(f"✅ Plots generated successfully.\n📂 Loss plot: {loss_plot_path}")
    except FileNotFoundError:
        print("❌ Error: Gnuplot is not installed or not in PATH.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Gnuplot execution failed:\n{e.stderr}")


    
