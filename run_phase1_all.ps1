# =============================================================================
# Phase I Full Batch Runner
# Runs AI categorization on all remaining Phase I items (offset 200+)
# 100 items per batch, 3-minute cooldown between batches
# =============================================================================

$ErrorActionPreference = "Continue"
$baseUrl = "http://localhost:8000"
$logFile = "phase1_batch_log.txt"
$startOffset = 200
$batchSize = 100
$totalItems = 7968
$cooldownSec = 180   # 3 minutes between batches
$vectorTopK = 10
$pollIntervalSec = 10

# Calculate batches
$endOffset = $totalItems
$totalBatches = [math]::Ceiling(($endOffset - $startOffset) / $batchSize)

function Log($msg) {
    $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path $logFile -Value $line
}

Log "============================================="
Log "PHASE I FULL BATCH RUN"
Log "============================================="
Log "Start offset:    $startOffset"
Log "Batch size:      $batchSize"
Log "Total items:     $totalItems"
Log "Total batches:   $totalBatches"
Log "Cooldown:        ${cooldownSec}s (3 min)"
Log "Vector top-K:    $vectorTopK"
Log "============================================="
Log ""

# Counters
$globalStart = Get-Date
$completedBatches = 0
$totalProcessed = 0
$totalHigh = 0
$totalMedium = 0
$totalLow = 0
$totalError = 0
$totalWritten = 0
$failedBatches = @()

for ($offset = $startOffset; $offset -lt $endOffset; $offset += $batchSize) {
    $batchNum = $completedBatches + 1
    $remaining = $totalBatches - $completedBatches
    $itemsInBatch = [math]::Min($batchSize, $endOffset - $offset)

    Log "--- Batch $batchNum / $totalBatches (offset=$offset, limit=$itemsInBatch) ---"

    # Submit batch
    try {
        $body = @{offset=$offset; limit=$itemsInBatch; vector_top_k=$vectorTopK} | ConvertTo-Json
        $response = Invoke-RestMethod -Uri "$baseUrl/api/phase1/batch/run" -Method POST -ContentType "application/json" -Body $body
        $jobId = $response.job_id
        Log "  Job submitted: $jobId"
    } catch {
        Log "  ERROR submitting batch: $_"
        $failedBatches += $offset
        Log "  Waiting cooldown before retry..."
        Start-Sleep -Seconds $cooldownSec
        continue
    }

    # Poll until done
    $batchStart = Get-Date
    $maxWaitSec = 600  # 10 min max per batch
    do {
        Start-Sleep -Seconds $pollIntervalSec
        try {
            $status = Invoke-RestMethod -Uri "$baseUrl/api/batch/$jobId/status" -Method GET
        } catch {
            Log "  Poll error: $_ — retrying..."
            continue
        }
        $elapsed = ((Get-Date) - $batchStart).TotalSeconds
    } while (($status.status -eq "running" -or $status.status -eq "queued") -and $elapsed -lt $maxWaitSec)

    $batchDuration = [math]::Round(((Get-Date) - $batchStart).TotalSeconds, 1)

    if ($status.status -eq "done") {
        $completedBatches++
        $batchTotal = [int]$status.total
        $batchHigh = [int]$status.high
        $batchMedium = [int]$status.medium
        $batchLow = [int]$status.low
        $batchErr = [int]$status.error
        $batchWritten = [int]$status.written_to_excel

        $totalProcessed += $batchTotal
        $totalHigh += $batchHigh
        $totalMedium += $batchMedium
        $totalLow += $batchLow
        $totalError += $batchErr
        $totalWritten += $batchWritten

        $globalElapsed = ((Get-Date) - $globalStart).TotalMinutes
        $avgPerBatch = [math]::Round($globalElapsed / $completedBatches, 1)
        $etaMin = [math]::Round($avgPerBatch * ($totalBatches - $completedBatches), 0)
        $etaHours = [math]::Round($etaMin / 60, 1)

        Log "  DONE in ${batchDuration}s — items=$batchTotal H=$batchHigh M=$batchMedium L=$batchLow E=$batchErr written=$batchWritten"
        Log "  Progress: $completedBatches/$totalBatches batches | $totalProcessed items | ETA: ${etaMin}min (${etaHours}h)"
    } elseif ($status.status -eq "error") {
        $errMsg = [string]$status.error
        if ($errMsg.Length -gt 150) { $errMsg = $errMsg.Substring(0, 150) + "..." }
        Log "  ERROR: $errMsg"
        $failedBatches += $offset
        $completedBatches++  # count as attempted
    } else {
        Log "  TIMEOUT after ${batchDuration}s — status=$($status.status)"
        $failedBatches += $offset
        $completedBatches++
    }

    # Cooldown (skip on last batch)
    if ($offset + $batchSize -lt $endOffset) {
        Log "  Cooling down ${cooldownSec}s..."
        Start-Sleep -Seconds $cooldownSec
    }
}

# Final summary
$globalDuration = ((Get-Date) - $globalStart).TotalMinutes
Log ""
Log "============================================="
Log "       PHASE I BATCH RUN COMPLETE"
Log "============================================="
Log "Total duration:  $([math]::Round($globalDuration, 1)) min ($([math]::Round($globalDuration/60, 1)) hours)"
Log "Batches:         $completedBatches / $totalBatches"
Log "Items processed: $totalProcessed"
Log "High confidence: $totalHigh"
Log "Medium conf:     $totalMedium"
Log "Low conf:        $totalLow"
Log "Errors:          $totalError"
Log "Written to Excel:$totalWritten"
if ($failedBatches.Count -gt 0) {
    Log "Failed offsets:  $($failedBatches -join ', ')"
}
Log "============================================="
