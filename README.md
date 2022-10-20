# stable-diffusion-webui-lite

    A lightweight version of stable-diffusion-webui, for easy public serving through ngrok tunnel from local PC ;)

----

# DEPRECATED, DO NOT USE!!

We do code cleanify on AUTOMATIC1111's [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui), making modules hierarchy and tidy.
Concerning about code clarity and avoiding compatiblity hells, we so far only support **Windows** platform with an RXT 3060 ;)  

⚠ This repo is for public web serving and painiting parameters exploring, hence unsafe functions are removed or behaviour changed.  
⚠ For personal experiments or professional research, you should probobly use the original one :(  

Visit my service for quick experience => [https://kahsolt.pythonanywhere.com/stable-diffusion-webui-lite](https://kahsolt.pythonanywhere.com/stable-diffusion-webui-lite)


### Features

- [x] basic txt2img, img2img, textual-inverse
- [ ] control variable experiments
- [ ] recommend prompt words
- [ ] show model stats (loss, gradients, etc..)


### Applications

- txt2img: 
- img2img: 
- sketchpad: upload & edit your image, generate  recommend prompts
- control variable: genarate a series of images with only one parameter changing within given range while others fixed
- postprocess
  - face restore: apply models for better face quality
  - super resoultion: apply models for higher resolution


### How to deploy my own

This project mainly consists of three workers:

- webui app server
- ngrok client & monitor website
- daemon

#### Step 1: setup stable-diffusion toolchains

- install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and [Git](https://gitforwindows.org/)
- clone this repo `git clone https://github.com/Kahsolt/stable-diffusion-webui-lite`
- run `install.cmd` to setup runtime environment
- place your stable-diffusion checkpoint file (like `model.pth`) under `models/stable-diffusion` folder 
- run `webui.cmd` to lauch the webui app

ℹ Check it: Now you should be able to visit the **local** service at `http://localhost:7860/`

#### Step 2: expose local service to public network

If you already have a running server with an IPv4 address on public network, you can surely skip this step ;)  
Otherwise, you need two more tools: 1. a frp-like tool `ngrok` for NAT tunnels to bridge local service to public; 2. a panel webserver like `pythonanywhere` for a fixed domain name, step them up as follows:

- register a [ngrok](https://ngrok.com/) account  (free plan is ok)
  - download and unzip the `ngrok.exe` executable to `ngrok` folder 
  - create a `API_KEY`  in your dashboard and save it to file `ngrok/api_key.txt` (create by hand)
  - run `python ngrok/api_ngrok.py` to test your key is ok

ℹ Check it: Now **everyone** should be able to visit your service through ngrok's tunnel like `http://xxxx-xx-xxx-.xx.ngrok.com/` on public network

However, the free plan of ngrok won't give you a fixed domain name, every time you restart ngrok manually or due to network errors, the service url will change, which makes the service rather unstable. Hence we need a fix domain name to track ngrok's dynamic-generated domain name. Luckily, `pythonanywhere.com` will do!

- register a [pythonanywhere](https://www.pythonanywhere.com) account  (free plan is ok)
  - now you have a free domain name like `<username>.pythonanywhere.com`
  - create a `API_KEY` 
  - 

ℹ  Check it: Now everyone should be able to visit your service **redirected** from your public pythonanywhere site like `http://<username>.pythonanywhere.com/stable-diffusion-webui-lite`

#### Step 3: start stable serving 

Once you've done the steps above without errors, you can safely shut them down, then try this all-in-one launcher script:

- run `start.cmd`

This will start both **webui server** and **ngrok client**, together with an extra daemon service to monitor, if either is dead unexpectly, the daemon will try to restart it automatically.

Don't forget that your pythonanywhere site will also track your ngrok status.

ℹ Now, everything's perfectly done! Taker your time~

Make Setu Great Again! 😀


#### Acknowledgment

Greatest thanks all contributing developers concerned with Stable-Diffusion.

- stable-diffusion-webui - https://github.com/AUTOMATIC1111/stable-diffusion-webui
- Stable Diffusion - https://github.com/CompVis/stable-diffusion, 
- Taming Transformers - https://github.com/CompVis/taming-transformers
- k-diffusion - https://github.com/crowsonkb/k-diffusion.git
- GFPGAN - https://github.com/TencentARC/GFPGAN.git
- CodeFormer - https://github.com/sczhou/CodeFormer
- ESRGAN - https://github.com/xinntao/ESRGAN
- SwinIR - https://github.com/JingyunLiang/SwinIR
- LDSR - https://github.com/Hafiidz/latent-diffusion
- Ideas for optimizations - https://github.com/basujindal/stable-diffusion
- Doggettx - Cross Attention layer optimization - https://github.com/Doggettx/stable-diffusion, original idea for prompt editing.
- Rinon Gal - Textual Inversion - https://github.com/rinongal/textual_inversion (we're not using his code, but we are using his ideas).
- Idea for SD upscale - https://github.com/jquesnelle/txt2imghd
- Noise generation for outpainting mk2 - https://github.com/parlance-zz/g-diffuser-bot
- CLIP interrogator idea and borrowing some code - https://github.com/pharmapsychotic/clip-interrogator
- Initial Gradio script - posted on 4chan by an Anonymous user. Thank you Anonymous user.
- DeepDanbooru - interrogator for anime diffusors https://github.com/KichangKim/DeepDanbooru
- (You)

----

by Armit
2022/10/10
