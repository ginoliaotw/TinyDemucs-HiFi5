import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import os
import argparse
from tqdm import tqdm

try:
    from demucs.pretrained import get_model as get_demucs_model
    from demucs.apply import apply_model
except ImportError:
    print("Error: Please install demucs via 'pip install demucs'")
    exit(1)

# Import the architecture we defined for the DSP
from tinydemucs_dsp import TinyDemucs

def get_teacher_model(device):
    """Loads the heavy, pre-trained Demucs model to act as the Teacher."""
    print("Loading Teacher Model (htdemucs)...")
    teacher = get_demucs_model(name='htdemucs')
    teacher.to(device)
    teacher.eval() # Teacher should not learn
    return teacher

def load_audio_segment(file_path, sr=44100, segment_length=221184):
    """
    Loads a random segment of audio for training.
    segment_length is 221184 (exactly 108 blocks of 2048), ensuring it's divisible by 16 
    so the 4 structural U-Net downsamplings don't cause dimension mismatch.
    """
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, sr)
    
    # Mono to stereo if needed
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
        
    # Pick a random 5-second chunk if file is long enough
    total_frames = waveform.shape[1]
    if total_frames > segment_length:
        start = torch.randint(0, total_frames - segment_length, (1,)).item()
        waveform = waveform[:, start:start+segment_length]
    else:
        # Pad if too short
        pad_size = segment_length - total_frames
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        
    return waveform.unsqueeze(0) # Add batch dimension: [B, C, T]

def train_kd(args):
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # 1. Initialize Student and Teacher
    teacher = get_teacher_model(device)
    student = TinyDemucs(channels=2).to(device)
    
    # 2. Setup Optimizer and Loss
    optimizer = optim.Adam(student.parameters(), lr=1e-3)
    criterion = nn.L1Loss() # L1 is often better for audio than MSE
    
    # Gather files (Recursive search for nested albums)
    import glob
    import random
    if os.path.isdir(args.input):
        audio_files = glob.glob(os.path.join(args.input, "**/*.wav"), recursive=True) + \
                      glob.glob(os.path.join(args.input, "**/*.mp3"), recursive=True)
    else:
        audio_files = [args.input]
        
    if not audio_files:
        print(f"Error: No audio files found in {args.input}")
        return
        
    print(f"--- Starting Knowledge Distillation ---")
    print(f"Training on {len(audio_files)} files from: {args.input}")
    print(f"Epochs: {args.epochs}")
    
    student.train()
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        
        for step in tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs}"):
            
            # Step A: Get raw mixed audio (randomly select a file)
            selected_file = random.choice(audio_files)
            mixed_audio = load_audio_segment(selected_file).to(device)
            
            # Step B: Get "Ground Truth" from Teacher (Demucs)
            with torch.no_grad():
                # apply_model handles the complex chunking for Demucs
                teacher_output = apply_model(teacher, mixed_audio, shifts=1, split=True, overlap=0.25)
                # target_sources shape: (batch, sources, channels, time)
                # sources order defined in teacher.sources (usually: drums, bass, other, vocals)
                vocal_idx = teacher.sources.index('vocals')
                teacher_vocals = teacher_output[:, vocal_idx, :, :]
            
            # Step C: Student attempts to separate
            optimizer.zero_grad()
            student_vocals = student(mixed_audio)
            
            # Step D: Calculate Distillation Loss
            # We want the student to mimic the teacher's output perfectly
            loss = criterion(student_vocals, teacher_vocals)
            
            # Step E: Backpropagation
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / args.steps_per_epoch:.4f}")
        
    # Save the trained weights
    save_path = os.path.join(args.output_dir, "tinydemucs_best.pth")
    torch.save(student.state_dict(), save_path)
    print(f"--- Training Complete ---")
    print(f"Trained student weights saved to: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Knowledge Distillation for TinyDemucs")
    parser.add_argument('-i', '--input', type=str, required=True, help="Input WAV to train on (Dummy data)")
    parser.add_argument('-e', '--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('-s', '--steps_per_epoch', type=int, default=10, help="Simulated dataset size per epoch")
    parser.add_argument('-o', '--output_dir', type=str, default=".", help="Directory to save weights")
    args = parser.parse_args()
    
    train_kd(args)
