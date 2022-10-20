import os
import sys
import subprocess
from importlib.util import find_spec
import platform


GITHUB_REPOS = [
  # ('author', 'repo')
  ('CompVis',    'stable-diffusion'   ),   # image genegration
  ('CompVis',    'taming-transformers'),   # image genegration
  ('crowsonkb',  'k-diffusion'        ),   # image genegration
  ('sczhou',     'CodeFormer'         ),   # face restore
  ('salesforce', 'BLIP'               ),   # CV-NLP comprehension
]
REPOS = {
  # 'repo': 'commit_hash'
  'stable-diffusion':    os.environ.get('STABLE_DIFFUSION_COMMIT_HASH',    '69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc'),
  'taming-transformers': os.environ.get('TAMING_TRANSFORMERS_COMMIT_HASH', '24268930bf1dce879235a7fddd0b2355b84d7ea6'),
  'k-diffusion':         os.environ.get('K_DIFFUSION_COMMIT_HASH',         'f4e99857772fc3a126ba886aadf795a332774878'),
  'CodeFormer':          os.environ.get('CODEFORMER_COMMIT_HASH',          'c5b4593074ba6214284d6acd5f1719b6c5d739af'),
  'BLIP':                os.environ.get('BLIP_COMMIT_HASH',                '48211a1594f1321b00f14c9f7a5b4813144b2fb9'),
}
PACKAGES = {
  # 'level': { 'package': 'pip install {args}' }
  'base': {
    'torch':  'torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113',
  }, 
  'required': {
    'clip':    'git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1',
    'gfpgan':  'git+https://github.com/TencentARC/GFPGAN.git@8d2447a2d918f8eba5a4a01463fd48e45126a379',
  },
  'optional': {
    'xformers':     ('https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/a/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl' if platform.system() == 'Windows' else 'xformers') + ' functorch==0.2.1',
    'deepdanbooru': 'git+https://github.com/KichangKim/DeepDanbooru.git@edf73df4cdaeea2cf00e9ac08bd8a9026b7a7b26#egg=deepdanbooru[tensorflow] tensorflow==2.10.0 tensorflow-io==0.27.0',
  },
}


GIT = os.environ.get('GIT', 'git')
PYTHON = sys.executable
PIP_MIRROR = 'https://pypi.tuna.tsinghua.edu.cn/simple'

BASE_PATH = os.path.dirname(os.path.abspath(__name__))
REPO_PATH = os.path.join(BASE_PATH, 'repos')

REPO_PATHS = {
  # 'stable-diffusion': './repos/stable-diffusion'
  repo: os.path.join(REPO_PATH, repo) for repo in REPOS
}


def sh(cmd:str, echo=True, raise_on_error=True) -> str:
  r = subprocess.run(cmd, capture_output=True, shell=True, text=True, encoding='utf-8', errors='ignore')
  code = r.returncode

  if code == 0:
    stdout = r.stdout.strip()
    if echo and stdout: print(stdout)
    return stdout
  else: 
    stdout = r.stdout.strip() or '<empty>'
    stderr = r.stderr.strip() or '<empty>'
    msg = (f'Error running command\n' + 
           f'   command: {cmd}\n' +
           f'   error code: {code}\n' +
           f'   stdout: {stdout}\n' +
           f'   stderr: {stderr}\n')
    if raise_on_error: raise RuntimeError(msg)
    else: print(msg)


def git_repo(author, repo):
  dp = REPO_PATHS[repo]

  if not os.path.exists(dp):
    url = f'https://github.com/{author}/{repo}.git'
    print(f'Cloning {url} into {dp}...')
    sh(f'"{GIT}" clone "{url}" "{dp}"')

  commit = REPOS.get(repo)
  if commit is not None:
    print(f'Get current commit hash for {repo}')
    cur_commit = sh(f'"{GIT}" -C {dp} rev-parse HEAD', echo=False, raise_on_error=False)
    print(f'Current commit hash: {cur_commit}')
    if cur_commit == commit: return
    
    print(f'Fetching updates for {repo}...')
    sh(f'"{GIT}" -C {dp} fetch')
    print(f'Checking out commint to hash: {commit}')
    sh(f'"{GIT}" -C {dp} checkout {commit}')


def install_repos():
  os.makedirs(REPO_PATH, exist_ok=True)

  for author, repo in GITHUB_REPOS:
    git_repo(author, repo)


def is_installed(package) -> bool:
  try: return find_spec(package) is not None
  except ModuleNotFoundError: return False


def pip_install(args, raise_on_error=True):
  pip_args = f'--prefer-binary --progress-bar off {f"-i {PIP_MIRROR}" if PIP_MIRROR else ""}'
  return sh(f'"{PYTHON}" -m pip install {args} {pip_args}', raise_on_error=raise_on_error)


def install_packages():
  # upgrade pip itself if possible
  pip_install('--upgrade pip', raise_on_error=False)

  # install 'base' packages (carefully check each)
  base_packages = PACKAGES['base']
  if False in [is_installed('torch'), is_installed('torchvision')]:
    print("Installing torch and torchvision")
    pip_install(base_packages['torch'])

  # test cuda is_available
  try:
    sh(f'"{PYTHON}" -c "import torch; assert torch.cuda.is_available()"')
  except RuntimeError:
    print('Warning: cuda is not avaliable in venv python') 

  # install 'required' packages
  for package, args in PACKAGES['required'].items():
    if not is_installed(package):
      print(f'Install pip package {package!r}')
      pip_install(args)

  # install 'optional' packages
  for package, args in PACKAGES['optional'].items():
    if not is_installed(package):
      print(f'Install pip package {package!r}')
      pip_install(args, raise_on_error=False)

  # install from 'requirements.txt'
  print('Install requirements.txt for stable-diffusion-webui-lite')
  pip_install(f"-r requirements.txt")


if __name__ == '__main__':
  print(f"Python built: {sys.version}")
  commit = sh(f"{GIT} rev-parse HEAD", echo=False, raise_on_error=False)
  print(f"commit hash: {commit}")

  print('>> Install git repos ...')
  install_repos()

  print('>> Install python packages ...')
  install_packages()
