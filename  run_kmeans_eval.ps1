Set-Location "d:\PycharmProjects\Face-Recognition"

# 关键：不要让原生命令的 stderr 触发 Stop
$ErrorActionPreference = "Continue"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $false
}

# 可选：减少 TF 日志噪音
$env:TF_CPP_MIN_LOG_LEVEL = "2"

$root = (Get-Location).Path
$srcDir = Join-Path $root "src"
$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$datasets = @(
    @{ name = "Alan_Ball";     video = "..\data\train_data\Alan_Ball\%04d.jpg" },
    @{ name = "Nancy_Sinatra"; video = "..\data\train_data\Nancy_Sinatra\%04d.jpg" },
    @{ name = "Peter_Gilmour";     video = "..\data\train_data\Peter_Gilmour\%04d.jpg" }
)

Set-Location $srcDir

foreach ($ds in $datasets) {
    Remove-Item "..\data\clustering_gallery.pkl" -ErrorAction SilentlyContinue
    $logFile = Join-Path $logDir ("kmeans_{0}.txt" -f $ds.name)
    Write-Host "Running dataset: $($ds.name)"

    # 用 cmd 做重定向，最稳
    cmd /c "python -m cvproj_exc.training --mode cluster --video ""$($ds.video)"" > ""$logFile"" 2>&1"
}

# 提取 iter/sse
$rows = @()
Get-ChildItem -Path $logDir -Filter "kmeans_*.txt" | ForEach-Object {
    $dataset = $_.BaseName -replace "^kmeans_", ""
    Select-String -Path $_.FullName -Pattern "iter\s+(\d+):\s+([0-9.]+)" | ForEach-Object {
        $m = [regex]::Match($_.Line, "iter\s+(\d+):\s+([0-9.]+)")
        $rows += [pscustomobject]@{
            dataset = $dataset
            iter    = [int]$m.Groups[1].Value
            sse     = [double]$m.Groups[2].Value
        }
    }
}

if ($rows.Count -eq 0) {
    Write-Host "没有提取到任何 iter/sse。请先看日志最后 80 行定位训练问题："
    Get-ChildItem -Path $logDir -Filter "kmeans_*.txt" | ForEach-Object {
        Write-Host "n===== $($_.Name) ====="
        Get-Content $_.FullName -Tail 80
    }
    exit 1
}

$rows = $rows | Sort-Object dataset, iter
$csvPath = Join-Path $logDir "kmeans_objective.csv"
$rows | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8
$rows | Format-Table -AutoSize
Write-Host "nCSV saved to: $csvPath"

# 画图
$pngPath = Join-Path $logDir "kmeans_objective.png"
@"
import pandas as pd
import matplotlib.pyplot as plt

csv_path = r"$csvPath"
png_path = r"$pngPath"

df = pd.read_csv(csv_path)
for name, g in df.groupby("dataset"):
    g = g.sort_values("iter")
    plt.plot(g["iter"], g["sse"], marker="o", label=name)

plt.xlabel("Iteration")
plt.ylabel("K-means objective (SSE)")
plt.title("Objective over iterations")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(png_path, dpi=180)
print("PNG saved to:", png_path)
"@ | python -

Write-Host "Done."