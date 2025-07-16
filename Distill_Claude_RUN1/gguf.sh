#!/bin/bash

#NOTE: BEFORE RUNNING SET VENV AND CALL IT `llama-venv` & YOU HAVE TO USE PYTHON 3.10


# === Things to do ===
# [x] git clone llama.cpp ; cd into it
# [x] make the file for accelerators -> set metal or whatever
# [x] set venv -> need to mention this in gh docs
# [x] install pip dependencies -> hf_hub, git+transformers, protobuf, sentencepiece, torch & llama_cpp ; better to have requirements.txt or define header
# [x] download model, then store in sub dir and no symlinks
# [x] convert to gguf
# [x] 4B quantize
# [x] build the 4B model
# [x] create the python script to use local model





set -e

# === Model Config ===
MODEL_REPO="MebinThattil/FT-LlaMA-135-Claude_Sonnet_4-Distill-RUN1"
GGUF_OUT="LlaMA-135M-Claude_Sonnet_4.gguf"
GGUF_QUANT="LlaMA-135M-Claude_Sonnet_4-q4.gguf"
N_CTX=2048
BUILD_DIR="build"
SAVED_DIR_NAME_HF="LlaMA-135M-Claude_Sonnet_4"

# === Detect Platform ===
OS="$(uname -s)"
ARCH="$(uname -m)"
echo "Detected OS: $OS & Arch: $ARCH"
echo "If this is wrong, manually override on line 33"

# === Dependencies list & script path====
REQUIREMENTS="torch huggingface_hub git+https://github.com/huggingface/transformers.git@main protobuf sentencepiece huggingface_hub[cli]"
RAW_URL="https://raw.githubusercontent.com/mebinthattil/template_llama_chat_python/main/chatapp.py"


#====================



# 1. === Setup llama.cpp ===
if [ ! -d "llama.cpp" ]; then
    echo "Cloning llama.cpp"
    git clone https://github.com/ggerganov/llama.cpp.git --depth 1
else
    echo "llama.cpp exists. Skipping clone."
fi
cd llama.cpp




# 2. === Detect best accelerator depending on hardware used to run ===

mkdir -p $BUILD_DIR
cd $BUILD_DIR
BACKEND_FLAGS="-DLLAMA_BLAS=on"

if [[ "$OS" == "Darwin" ]]; then
    echo "On Mac, enabling Metal 🤘XD"
    BACKEND_FLAGS="-DLLAMA_METAL=on"
elif command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA enabling CUDA"
    BACKEND_FLAGS="-DLLAMA_CUBLAS=on"
else
    echo "No GPU using CPU"
fi

echo "Building llama.cpp with: $BACKEND_FLAGS"
cmake .. -DCMAKE_BUILD_TYPE=Release $BACKEND_FLAGS
cmake --build . --config Release

cd ../..






# 3. === Set venv ===

    #first check if its present, else exit. I'll mention to activiate env in gh readme
if [ ! -d "llama-venv" ]; then
    echo "Does not look like you have setup and activated the venv required. Refer to https://github.com/mebinthattil/Model_Quantize_Pipeline/blob/main/README.md"
    exit 1
fi

    # Now confirmed if venv is present, so check OS type and activiate
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then 
    source llama-venv/Scripts/activate
else 
    source llama-venv/bin/activate
fi

echo "Virtual environment activated."





# 4. === Install dependencies ===
#iterate through the prev defined requirements var and pip install
for package in $REQUIREMENTS; do
    echo "Installing $package"
    pip install "$package"
    
    # Optional: check if the install succeeded
    if [ $? -ne 0 ]; then
        echo "Failed to install $package"
        exit 1
    
    echo "$package  was installed sucessfully."

    fi
done

echo "All packages installed successfully."


# === 5. Download model and store in subdirectory ===
echo "Downloading model from Hugging Face repo: $MODEL_REPO"


python3 - <<EOF
from huggingface_hub import snapshot_download

repo_id = "$MODEL_REPO"
snapshot_download(
    repo_id=repo_id,
    local_dir="$SAVED_DIR_NAME_HF",
    local_dir_use_symlinks=False
)
EOF

if [ $? -ne 0 ]; then
    echo "Error: model download from HF failed. Please ensure you are logged into Hugging Face and the model exists: $MODEL_REPO"
    echo "You could also checkout the HF docs: https://huggingface.co/docs/huggingface_hub/main/en/guides/cli"
    exit 1
fi



# === Install python bindings ===
echo "Installing llama-cpp-python"
if [[ "$BACKEND_FLAGS" == *"METAL"* ]]; then
    CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
elif [[ "$BACKEND_FLAGS" == *"CUBLAS"* ]]; then
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
else
    pip install llama-cpp-python
fi


# === Convert to GGUF ===
echo "Converting to GGUF"
python3 llama.cpp/convert_hf_to_gguf.py $SAVED_DIR_NAME_HF --outfile $GGUF_OUT

# === Quantize ===
echo "Quantizing to Q4_0"
./llama.cpp/build/bin/llama-quantize $GGUF_OUT $GGUF_QUANT Q4_0


# === Done, now download my script and run ===
#remember lot of things hardcoded here, it might break easily
echo "Downloading python script to run from GitHub"
curl -o chatapp.py "$RAW_URL"

if [ $? -ne 0 ] || [ ! -f chatapp.py ]; then
    echo "Failed to download chatapp.py"
    exit 1
fi

echo "Running chatapp.py..."
python chatapp.py $GGUF_QUANT

