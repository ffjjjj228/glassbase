
import re
import sys
import pathlib
from dataclasses import dataclass, field
from typing import List, Optional

from pyhocon import ConfigFactory, ConfigTree

from src.torchutils.common import get_free_port, apply_modifications
from src.torchutils.typed_args import TypedArgs, add_argument


def is_valid_domain(value):
    pattern = re.compile(
        r'^(([a-zA-Z]{1})|([a-zA-Z]{1}[a-zA-Z]{1})|'
        r'([a-zA-Z]{1}[0-9]{1})|([0-9]{1}[a-zA-Z]{1})|'
        r'([a-zA-Z0-9][-_.a-zA-Z0-9]{0,61}[a-zA-Z0-9]))\.'
        r'([a-zA-Z]{2,13}|[a-zA-Z0-9-]{2,30}.[a-zA-Z]{2,3})$'
    )
    return True if pattern.match(value) else False


def is_valid_ip(str):
    p = re.compile('^((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$')
    return True if p.match(str) else False



@dataclass
class Args(TypedArgs):
    output_dir: str = field(default_factory=lambda: add_argument("-o", "--output-dir", default=""))
    conf: str = field(default_factory=lambda: add_argument("--conf", default=""))
    modifications: List[str] = field(default_factory=lambda: add_argument("-M", nargs='+', help="list"))

    world_size: int = field(default_factory=lambda: add_argument("--world-size", default=1))
    dist_backend: str = field(default_factory=lambda: add_argument("--dist-backend", default="nccl"))
    dist_url: Optional[str] = field(default_factory=lambda: add_argument("--dist-url", default=None))
    node_rank: int = field(default_factory=lambda: add_argument("--node-rank", default=0))


def _auto_output_dir(conf_path: str) -> pathlib.Path:
    conf_name = pathlib.Path(conf_path).stem
    output_root = pathlib.Path("output")
    output_root.mkdir(exist_ok=True)

    max_num = 0
    for d in output_root.iterdir():
        if d.is_dir() and d.name.startswith(f"{conf_name}_"):
            suffix = d.name[len(conf_name) + 1:]
            if suffix.isdigit():
                max_num = max(max_num, int(suffix))

    run_dir = f"{conf_name}_{max_num + 1:02d}"
    return output_root / run_dir


def _next_output_dir(path: pathlib.Path) -> pathlib.Path:
    if not path.exists():
        return path
    output_root = path.parent
    stem = path.name
    max_num = 0
    for d in output_root.iterdir():
        if d.is_dir() and d.name.startswith(f"{stem}_"):
            suffix = d.name[len(stem) + 1:]
            if suffix.isdigit():
                max_num = max(max_num, int(suffix))
    return output_root / f"{stem}_{max_num + 1:02d}"


def get_args(argv=sys.argv):
    args, _ = Args.from_known_args(argv)
    user_specified = bool(pathlib.Path(args.output_dir).name)

    if user_specified:
        args.output_dir = _next_output_dir(pathlib.Path(args.output_dir))
    else:
        args.output_dir = _auto_output_dir(args.conf)

    if args.dist_url is None:
        args.dist_url = f"tcp://127.0.0.1:{get_free_port()}"
    elif is_valid_domain(args.dist_url) or is_valid_ip(args.dist_url):
        args.dist_url = f"tcp://{args.dist_url}:{get_free_port()}"

    args.conf: ConfigTree = ConfigFactory.parse_file(args.conf)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    apply_modifications(modifications=args.modifications, conf=args.conf)

    return args
