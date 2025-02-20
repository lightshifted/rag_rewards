How to install Conda in ssh endpoint. Here's the step-by-step process:

1. First, connect to your endpoint via SSH:
```bash
ssh username@hostname
```

2. Once connected, download the Miniconda installer script:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

3. Make the installer executable:
```bash
chmod +x Miniconda3-latest-Linux-x86_64.sh
```

4. Run the installer:
```bash
./Miniconda3-latest-Linux-x86_64.sh -b
```
Note: The `-b` flag runs the installer in batch mode (non-interactive)

5. Initialize Conda in your shell:
```bash
source ~/miniconda3/bin/activate
```

6. Add Conda to your PATH permanently by adding it to your .bashrc:
```bash
echo 'export PATH="~/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

7. Verify the installation:
```bash
conda --version
```

