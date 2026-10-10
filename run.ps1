#Requires -RunAsAdministrator
# Run this script from an elevated PowerShell session.
# It shares matching USB webcams with WSL, then builds and starts Stereo Lab.

$ErrorActionPreference = 'Stop'
$hardwareId = '114d:23fb'
$projectDirectory = $PSScriptRoot

if (-not (Get-Command usbipd -ErrorAction SilentlyContinue)) {
    throw 'usbipd was not found. Install usbipd-win and try again.'
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Docker was not found. Install/start Docker Desktop and try again.'
}

Write-Host 'Reading connected USB devices from usbipd...'
$listing = @(& usbipd list 2>&1)
if ($LASTEXITCODE -ne 0) {
    throw "usbipd list failed: $($listing -join [Environment]::NewLine)"
}

# Parse only the Connected section. Persisted devices are not necessarily
# physically present, so they must not be selected for this startup.
$connectedDevices = @()
$inConnectedSection = $false
foreach ($line in $listing) {
    $text = [string]$line
    if ($text -match '^\s*Connected:\s*$') {
        $inConnectedSection = $true
        continue
    }
    if ($text -match '^\s*Persisted:\s*$') {
        break
    }
    if (-not $inConnectedSection) {
        continue
    }

    if ($text -match '^\s*(?<BusId>\d+-\d+(?:\.\d+)?)\s+(?<HardwareId>[0-9a-fA-F]{4}:[0-9a-fA-F]{4})\s+(?<Details>.+?)\s*$') {
        $busId = $Matches['BusId']
        $reportedHardwareId = $Matches['HardwareId']
        $details = $Matches['Details']
        $state = 'Unknown'
        if ($details -match '(?<State>Not shared|Shared|Attached)\s*$') {
            $state = $Matches['State']
        }
        if ($reportedHardwareId.ToLowerInvariant() -eq $hardwareId) {
            $connectedDevices += [PSCustomObject]@{
                BusId = $busId
                State = $state
                Details = $details
            }
        }
    }
}

if ($connectedDevices.Count -eq 0) {
    Write-Warning "No connected USB devices with hardware ID $hardwareId were found."
} else {
    foreach ($device in $connectedDevices) {
        Write-Host "Found $hardwareId at bus ID $($device.BusId) (state: $($device.State))."

        if ($device.State -eq 'Attached') {
            Write-Host "Already attached; leaving bus ID $($device.BusId) as-is."
            continue
        }

        if ($device.State -ne 'Shared') {
            $bindOutput = @(& usbipd bind --busid $device.BusId 2>&1)
            $bindExitCode = $LASTEXITCODE
            if ($bindExitCode -ne 0) {
                $bindMessage = $bindOutput -join [Environment]::NewLine
                if ($bindMessage -match '(?i)already\s+(shared|bound)') {
                    Write-Host "Bus ID $($device.BusId) is already shared."
                } else {
                    throw "usbipd bind failed for $($device.BusId): $bindMessage"
                }
            } else {
                Write-Host "Shared bus ID $($device.BusId) with WSL."
            }
        } else {
            Write-Host "Bus ID $($device.BusId) is already shared."
        }

        $attachOutput = @(& usbipd attach --wsl --busid $device.BusId --auto-attach 2>&1)
        $attachExitCode = $LASTEXITCODE
        if ($attachExitCode -ne 0) {
            $attachMessage = $attachOutput -join [Environment]::NewLine
            if ($attachMessage -match '(?i)already\s+attached') {
                Write-Host "Bus ID $($device.BusId) is already attached to WSL."
            } else {
                throw "usbipd attach failed for $($device.BusId): $attachMessage"
            }
        } else {
            Write-Host "Attached bus ID $($device.BusId) to WSL with auto-attach."
        }
    }
}

Push-Location $projectDirectory
try {
    Write-Host 'Building and starting Stereo Lab...'
    & docker compose `
        -f (Join-Path $projectDirectory 'docker-compose.yml') `
        -f (Join-Path $projectDirectory 'docker-compose.gpu.yml') `
        up -d --build
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Compose failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}

Write-Host 'Stereo Lab is starting. Open http://localhost:5000 when the container is ready.'
