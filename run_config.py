# contains main list of arguments for config
import argparse

def model_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=4, metavar='N')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5)
    parser.add_argument("-e", '--epochs', type=int, default=10, help="epochs")
    parser.add_argument("-s", '--steps_per_epoch', type=int, default=50, help="steps/epoch, validation at epoch end")
    parser.add_argument("-v", '--val_steps', type=int, default=8, help="validation steps/epoch")
    parser.add_argument('--data_path', type=str, default="data/raw", help="path to dataset")
    parser.add_argument('--num_frames', type=int, default=6, help="number of consecutive video frames (converted to attention maps)")
    parser.add_argument('--frame_hop', type=int, default=2, help="hop between each clip example in a video")
    parser.add_argument('--framerate', type=int, default=30, help="video fps")
    parser.add_argument('--framesize', type=int, default=256, help="scaled video frame dims (converted to attention maps)")
    parser.add_argument('--p_size', type=int, default=64, help="downsampled phasegram size")
    
    parser.add_argument('--autocontrast', type=bool, default=False, help="automatic video contrast")
    parser.add_argument('--compress_audio', action="store_true", help="audio compression")

    parser.add_argument('--fft_len', type=int, default=256, help="size of fft")
    parser.add_argument('-a', '--hops_per_frame', type=int, default=8, help="num hops per frame (a)")
    parser.add_argument('--samplerate', type=int, default=16000, help="audio samplerate (dependent on dataset)")
    parser.add_argument('--normalize_fft', type=bool, default=True, help="normalize input fft by 1/n")
    parser.add_argument('--normalize_output_fft', type=bool, default=False, help="normalize output fft by 1/max(abs(fft))")
    parser.add_argument('--use_polar', type=bool, default=False, help="fft uses polar coordinates instead of rectangular")
    parser.add_argument('--noise_scalar', type=float, default=0.1, help="scale gaussian noise by N for data augmentation (applied to x)")

    parser.add_argument('--fc_size', type=int, default=4096, help="size of deep fully connected layer")
    parser.add_argument('--latent_chan', type=int, default=64, help="num latent conv channels at autoencoder ends")

    parser.add_argument('--cb_freq', type=int, default=100, help="wandb callback frequency in epochs")
    parser.add_argument('--max_clip_len', type=int, default=None, help="maximum clip length to load (speed up loading)")
    parser.add_argument('--split', type=float, default=0.8, help="train/val split")
    parser.add_argument('--saved_model', type=str, default=None, help="path to saved model state dict")
    parser.add_argument('--checkpoint', type=str, default=None, help="load model checkpoint")

    parser.add_argument('--cp_dir', type=str, default="checkpoints/", help="specify checkpoint save directory")
    parser.add_argument('--cp_load_opt', action="store_true", help="load checkpoint optimizer config")
    parser.add_argument('-c', action="store_true", help="auto-loads the last saved checkpoint")
    parser.add_argument('--no_save', action="store_true", help="disable saving")
    args = parser.parse_args()
    return args