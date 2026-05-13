# ===== 配置区 =====
$task       = "DualGate-Attention-G1-Eval-v3"
$scriptPath = "D:\humanoid_locomotion\scripts\rsl_rl\evalute.py"
$speeds     = @(1.5, 1.2, 0.9, 0.6)

$experiments = @(
    @{
        Agent      = "mha_rsl_rl_cfg_entry_point"
        BaseDir    = "C:\Users\395\OneDrive\data\Attention-G1-v3\mha"
        Iterations = @(22400, 21600, 21000)
    },
    @{
        Agent      = "rsl_rl_cfg_entry_point"
        BaseDir    = "C:\Users\395\OneDrive\data\Attention-G1-v3\velocity_cnn"
        Iterations = @(23600, 23000, 22800)
    }
)

# ===== 主循环 =====
$totalCkpts = ($experiments | ForEach-Object { $_.Iterations.Count } | Measure-Object -Sum).Sum
$totalRuns  = $totalCkpts * $speeds.Count
$i = 0

foreach ($exp in $experiments) {
    $agent = $exp.Agent
    Write-Host ""
    Write-Host "################################################"
    Write-Host "# Agent: $agent"
    Write-Host "################################################"

    foreach ($iter in $exp.Iterations) {
        $ckpt = Join-Path $exp.BaseDir "model_$iter.pt"

        if (-not (Test-Path $ckpt)) {
            Write-Warning "Checkpoint 不存在，跳过: $ckpt"
            continue
        }

        $expTag = (Split-Path -Leaf $exp.BaseDir) + "_iter$iter"
        Write-Host ""
        Write-Host "----- Checkpoint: $expTag -----"

        foreach ($speed in $speeds) {
            $i++
            Write-Host "[$i/$totalRuns] >>> $expTag @ velocity=$speed"

            python $scriptPath `
                --task $task `
                --agent $agent `
                --headless `
                --checkpoint $ckpt `
                --velocity $speed

            if ($LASTEXITCODE -ne 0) {
                Write-Warning "Run failed (exit=$LASTEXITCODE): $expTag @ v=$speed"
            }
        }
    }
}

Write-Host ""
Write-Host "==================== All done ===================="
Write-Host "完成 runs: $i / $totalRuns"