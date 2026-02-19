# 在项目根目录执行
Set-Location "d:\PycharmProjects\Face-Recognition\src"
$dataset = "..\data\train_data\Peter_Gilmour\%04d.jpg"
$logDir = "..\logs\init_sensitivity"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# 1) 同一数据集重复跑 10 次（每次随机初始化不同）
for($r=1; $r -le 10; $r++){
    Remove-Item "..\data\clustering_gallery.pkl" -ErrorAction SilentlyContinue
    cmd /c "python -m cvproj_exc.training --mode cluster --video ""$dataset"" > ""$logDir\run_$r.txt"" 2>&1"
}

# 2) 提取每次的最终 SSE 和迭代数
$rows = @()
Get-ChildItem $logDir\run_*.txt | Sort-Object Name | ForEach-Object {
    $run = [int]([regex]::Match($_.BaseName, "run_(\d+)").Groups[1].Value)
    $lines = Select-String -Path $_.FullName -Pattern "iter\s+(\d+):\s+([0-9.]+)"
    if($lines.Count -gt 0){
        $last = $lines[-1].Line
        $m = [regex]::Match($last, "iter\s+(\d+):\s+([0-9.]+)")
        $rows += [pscustomobject]@{
            run = $run
            iters = $lines.Count
            final_sse = [double]$m.Groups[2].Value
        }
    }
}

$rows | Sort-Object run | Format-Table -AutoSize
$rows | Export-Csv "$logDir\summary.csv" -NoTypeInformation -Encoding UTF8

# 3) 统计波动（初始化敏感性）
$mean = ($rows.final_sse | Measure-Object -Average).Average
$sumsq = 0.0
$rows.final_sse | ForEach-Object { $sumsq += ($_ - $mean) * ($_ - $mean) }
$std = [math]::Sqrt($sumsq / [math]::Max(1, $rows.Count))
$cv = if($mean -ne 0){ $std / $mean } else { 0 }

"mean_final_sse = {0:N4}" -f $mean
"std_final_sse  = {0:N4}" -f $std
"cv(std/mean)   = {0:P2}" -f $cv
