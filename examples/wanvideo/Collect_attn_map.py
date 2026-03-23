import torch, os, imageio, argparse 
from torchvision.transforms import v2 
from einops import rearrange 
import lightning as pl
import pandas as pd
import sys 
from diffsynth_collect_attn import WanVideoPipeline_attn, ModelManager, load_state_dict 
import torchvision 
from PIL import Image
import math


def get_attention_mask(mask_name,num_frame=10,frame_size=400):
    attention_mask = torch.zeros((num_frame * frame_size, num_frame * frame_size),
                                 device="cpu")
    # TODO: fix hard coded mask
    if mask_name == "spatial":
        # print('spatial')
        pixel_attn_mask = torch.zeros_like(attention_mask, dtype=torch.float,
                                           device="cpu")
        block_size, block_thres = 128, frame_size
        num_block = math.ceil(num_frame * frame_size / block_size)
        for i in range(num_block):
            for j in range(num_block):
                if abs(i - j) < block_thres // block_size:
                    pixel_attn_mask[i * block_size: (i + 1) * block_size, j * block_size: (j + 1) * block_size] = 1
        attention_mask= pixel_attn_mask
        # attention_mask[-context_length:, :] = 1
        # attention_mask[:, -context_length:] = 1
        # attention_mask = torch.load(f"/data/home/xihaocheng/andy_develop/tmp_data/hunyuanvideo/I2VSparse/sparseattn/v5/mask_tensor/mask_spatial.pt", map_location="cpu")
    else:
        # print('temporal')
        pixel_attn_mask = torch.zeros_like(attention_mask, dtype=torch.float,
                                           device='cpu')
        block_size, block_thres = 128, frame_size
        num_block = math.ceil(num_frame * frame_size / block_size)
        for i in range(num_block):
            for j in range(num_block):
                if abs(i - j) < block_thres // block_size:
                    pixel_attn_mask[i * block_size: (i + 1) * block_size, j * block_size: (j + 1) * block_size] = 1
        pixel_attn_mask = pixel_attn_mask.reshape(frame_size, num_frame, frame_size, num_frame).permute(1, 0, 3,
                                                                                                        2).reshape(
            frame_size * num_frame, frame_size * num_frame)
        attention_mask  = pixel_attn_mask
        # attention_mask[-context_length:, :] = 1
        # attention_mask[:, -context_length:] = 1
        # attention_mask = torch.load(f"/data/home/xihaocheng/andy_develop/tmp_data/hunyuanvideo/I2VSparse/sparseattn/v5/mask_tensor/mask_temporal.pt", map_location="cpu")
    attention_mask = attention_mask
    return attention_mask


def classify_head(attn,frame_size):
    b,n,s=attn.shape[0],attn.shape[1],attn.shape[2]
    spatial_mask=get_attention_mask(mask_name='spatial',num_frame=s//frame_size,frame_size=frame_size)
    tempral_mask=get_attention_mask(mask_name='temporal',num_frame=s//frame_size,frame_size=frame_size)
    types=[]
    for i in range(n):
        attn_h=attn[:,i]#b s s
        # print(attn_h.min(),attn_h.max())
        w_s=(attn_h*spatial_mask).sum()
        w_t=(attn_h*tempral_mask).sum()
        print(w_s,w_t)
        if w_t<w_s*1.3:
            types.append(1)#spatial
        else:
            types.append(0)#temporal
    return types




class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480,
                 width=832):
        metadata = pd.read_csv(metadata_path)#
        self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        self.text = metadata["text"].to_list()

        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width

        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image

    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (
                num_frames - 1) * interval:
            reader.close()
            return None

        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        return frames

    def load_video(self, file_path):
        start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval,
                                                self.num_frames, self.frame_process)
        return frames

    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False

    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame

    def __getitem__(self, data_id):
        text = self.text[data_id]
        path = self.path[data_id]
        if self.is_image(path):
            video = self.load_image(path)
        else:
            video = self.load_video(path)
        data = {"text": text, "video": video, "path": path}
        return data

    def __len__(self):
        return len(self.path)


def shuffle_T_dimension(tensor):
    B, C, T, H, W = tensor.shape

    # 为每个样本和通道创建随机排列索引
    idx = torch.randperm(T)

    # 使用高级索引在T维度上重新排列
    # 注意：我们需要确保索引广播到正确的维度
    return tensor[:, :, idx, :, :]


class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models([text_encoder_path, vae_path])
        self.pipe = WanVideoPipeline_attn.from_model_manager(model_manager)

        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

    def test_step(self, batch, batch_idx):
        text, video, path = batch["text"][0], batch["video"], batch["path"][0]
        self.pipe.device = "cuda"
        if video is not None:
            prompt_emb = self.pipe.encode_prompt(text)
            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0]
            frames_latents = []
            print(video.shape)  # B C T H W

            b, c, t, h, w = video.shape
            for i in range(t):
                fram = video[:, :, i, :, :]
                fram = rearrange(fram, "B C H W -> B C 1 H W")
                frames_latents.append(self.pipe.encode_video(fram, **self.tiler_kwargs)[0])
            for i in range(16):
                video_ = shuffle_T_dimension(video)
                frames_latents.append(self.pipe.encode_video(video_, **self.tiler_kwargs)[0])

            data = {"latents": latents, "prompt_emb": prompt_emb, 'frames_latents': frames_latents}
            torch.save(data, path + ".tensors.pth")


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, steps_per_epoch):
        metadata = pd.read_csv(metadata_path)
        self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        print(len(self.path), "videos in metadata.")
        self.path = [i + ".tensors.pth" for i in self.path if os.path.exists(i + ".tensors.pth")]
        print(len(self.path), "tensors cached in metadata.")
        assert len(self.path) > 0

        self.steps_per_epoch = steps_per_epoch

    def __getitem__(self, index):
        data_id = torch.randint(0, len(self.path), (1,))[0]
        data_id = (data_id + index) % len(self.path)  # For fixed seed.
        path = self.path[data_id]
        data = torch.load(path, weights_only=True, map_location="cpu")
        return data

    def __len__(self):
        return self.steps_per_epoch


class ModelForAttnMap(torch.nn.Module):
    def __init__(
            self,
            dit_path,
            use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
    ):
        super().__init__()
        # self.temporal_lora_wd = temporal_lora_wd
        # self.train_temporal_lora = train_temporal_lora
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])

        self.pipe = WanVideoPipeline_attn.from_model_manager(model_manager)
        self.pipe.denoising_model().to("cuda")
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.freeze_parameters()
        # self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().eval()

    def forward(self, batch):
        # Data
        latents = batch["latents"].to("cuda")
        h, w=latents.shape[-2],latents.shape[-1]
        print('attn shape: ',latents.shape)
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][0].to("cuda")

        # Loss
        self.pipe.device = "cuda"
        noise = torch.randn_like(latents)
        timestep_id =torch.tensor([1]) #torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        # training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        noise_pred,all_attn = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, **prompt_emb, **extra_input,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )
        print('attention layer nums:', len(all_attn))

        all_head_type=[]
        for a in all_attn:
            print('attn shape',a.shape)#b n s s
            type_list=classify_head(a,frame_size=h*w//4)# size=num_head 
            all_head_type.append(type_list)
        return all_head_type,all_attn




def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")



    parser.add_argument(
        "--out_file_name",
        type=str,
        default="./",
        help="out_file_name",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=1,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Model structure to train. LoRA training or full training.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    args = parser.parse_args()
    return args




def analyse_attn_head(args):
    dataset = TensorDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.csv"),
        steps_per_epoch=args.steps_per_epoch,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model = ModelForAttnMap(
        dit_path=args.dit_path,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
    )
    for ind,batch in enumerate(dataloader):# should be all frames for a single video
        print(f'batch{ind}/{len(dataloader)}')
        all_head_type,all_attn=model(batch)#in the form of [[],[],,,,]; len(all_head_type): layer number of DiT blocks
        print(len(all_head_type),all_head_type)
        torch.save(all_head_type,os.path.join(args.dataset_path,'head_types.pt'))
        # torch.save(all_attn,os.path.join(args.dataset_path,'all_attn.pt'))


if __name__ == '__main__':
    args = parse_args()
    print(args)
    analyse_attn_head(args)
