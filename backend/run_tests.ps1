param(
    [string]$Path = "backend/app/tests",
    [switch]$Verbose
)

$argsList = @("-m", "pytest", $Path)
if ($Verbose) {
    $argsList += "-vv"
}

python @argsList
