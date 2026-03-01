import torch
import torch.nn as nn
import soundfile as sf
import numpy as np
import argparse
import os

class TinyDemucsBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        # Calculate padding to maintain correct alignment during downsampling
        padding = dilation
        # 1D Depthwise separable convolution with Dilation (Empty Convolution)
        self.conv_dw = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=False)
        self.conv_pw = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU() # DSP friendly

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.relu(x)
        x = self.conv_pw(x)
        x = self.relu(x)
        return x

class TinyDemucs(nn.Module):
    """
    An enhanced 4-layer DSP-friendly architecture.
    Now includes Dilated Convolutions and up to 256 channels for actual learning capacity.
    """
    def __init__(self, channels=2):
        super().__init__()
        # Deeper Encoder, more channels, increasing dilation (Receptive Field)
        self.enc1 = TinyDemucsBlock(channels, 32, stride=2, dilation=1)
        self.enc2 = TinyDemucsBlock(32, 64, stride=2, dilation=2)
        self.enc3 = TinyDemucsBlock(64, 128, stride=2, dilation=4)
        self.enc4 = TinyDemucsBlock(128, 256, stride=2, dilation=8)
        
        # Bottleneck
        self.bottleneck = TinyDemucsBlock(256, 256, stride=1, dilation=16)
        
        # Decoder (using ConvTranspose1d for upsampling)
        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU()
        )
        self.final = nn.Conv1d(16, channels, kernel_size=1)
        
    def forward(self, x):
        # Downsample
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        b = self.bottleneck(e4)
        
        # Upsample with Skip Connections
        d4 = self.dec4(b) + e3
        d3 = self.dec3(d4) + e2
        d2 = self.dec2(d3) + e1
        d1 = self.dec1(d2)
        
        # Directly predict the waveform. 
        # Using Tanh to keep the output safely bounded between [-1.0, 1.0] and prevent clipping.
        out = self.final(d1)
        
        return torch.tanh(out)

def mock_inference_stream(input_file, output_file, model, block_size=2048):
    """
    Simulates a C/C++ DSP inference loop.
    In a real Bluetrum DSP, DMA triggers a half-complete interrupt every 'block_size' samples.
    """
    print(f"--- Starting DSP Streaming Simulation ---")
    print(f"Input: {input_file}")
    print(f"Block Size (DMA Buffer): {block_size} samples")
    
    info = sf.info(input_file)
    sr = info.samplerate
    channels = info.channels
    
    # Pre-allocate output file
    sf.write(output_file, np.zeros((0, channels)), sr)
    
    model.eval()
    
    # Initialize overlap-add or history buffer here in strict C++.
    # For PyTorch block simulation, we pad the stream.
    
    with sf.SoundFile(input_file, 'r') as f_in, \
         sf.SoundFile(output_file, 'w', samplerate=sr, channels=channels) as f_out:
         
        total_blocks = info.frames // block_size
        block_idx = 0
        
        with torch.no_grad():
            for block in f_in.blocks(blocksize=block_size, dtype='float32'):
                if block.shape[0] < block_size:
                    # Pad last block
                    pad = np.zeros((block_size - block.shape[0], channels), dtype='float32')
                    block = np.vstack((block, pad))
                    
                if block.ndim == 1:
                    block = np.expand_dims(block, axis=1) # [T, C]
                    
                # Format to Tensor: [B=1, C=2, T=2048]
                tensor_in = torch.from_numpy(block).transpose(0, 1).unsqueeze(0)
                
                # DSP INNER KERNEL -------------------------
                tensor_out = model(tensor_in)
                # ------------------------------------------
                
                # Format to Audio: [T=2048, C=2]
                out_block = tensor_out.squeeze(0).transpose(0, 1).numpy()
                
                # Simulatd DAC Push
                f_out.write(out_block)
                
                block_idx += 1
                if block_idx % 2000 == 0:
                    print(f"Processed block {block_idx} / {total_blocks}")

    print(f"Finished processing. Output: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TinyDemucs HiFi-5 DSP Simulation")
    parser.add_argument('-i', '--input', type=str, required=True, help="Input WAV")
    parser.add_argument('-o', '--output', type=str, required=True, help="Output WAV")
    parser.add_argument('-b', '--block_size', type=int, default=2048, help="Buffer size")
    parser.add_argument('-w', '--weights', type=str, default=None, help="Path to trained weights (.pth)")
    args = parser.parse_args()

    model = TinyDemucs(channels=2)
    
    if args.weights and os.path.exists(args.weights):
        print(f"Loading trained weights from: {args.weights}")
        model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    else:
        print("WARNING: No weights provided or found. Using untrained (random) weights for architectural prototype purposes.")
        print("Please train the model using train_kd.py first.")
        
    mock_inference_stream(args.input, args.output, model, block_size=args.block_size)
