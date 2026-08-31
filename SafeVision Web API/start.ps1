param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 5000
)

$env:SAFEVISION_API_HOST = $HostAddress
$env:SAFEVISION_API_PORT = [string]$Port
Push-Location $PSScriptRoot
try {
    python -m waitress --host=$HostAddress --port=$Port wsgi:app
}
finally {
    Pop-Location
}
