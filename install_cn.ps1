Set-Location $PSScriptRoot

$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1

if (!(Test-Path -Path "venv")) {
    Write-Output  "Creating venv for python..."
    python -m venv venv
}
.\venv\Scripts\activate

#$Env:MPLLOCALFREETYPE = 1

Write-Output "Installing deps..."
pip install --upgrade -r requirements.txt -i https://mirror.baidu.com/pypi/simple

pip install opencv-python==4.6.0.66 opencv-python-headless==4.6.0.66 matplotlib==3.2.2 streamlit==1.14.1 streamlit-drawable-canvas==0.9.2

pip install git+https://github.com/openai/CLIP.git

Write-Output "Install completed"
Read-Host | Out-Null ;
