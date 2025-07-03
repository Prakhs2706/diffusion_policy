import zarr
import sys
import numpy as np

def print_attrs(name, obj):
    print(f"\n{name} (type: {type(obj)})")
    if hasattr(obj, 'attrs'):
        for k, v in obj.attrs.items():
            print(f"  attr[{k}]: {v}")

def inspect_zarr(path):
    print(f"Opening Zarr file: {path}")
    root = zarr.open(path, mode='r')

    def recurse(group, prefix=""):
        for name, item in group.items():
            full_name = f"{prefix}/{name}".lstrip("/")
            if isinstance(item, zarr.Group):
                print(f"\n📁 Group: {full_name}")
                print_attrs(full_name, item)
                recurse(item, prefix=full_name)
            elif isinstance(item, zarr.Array):
                print(f"\n📦 Array: {full_name}")
                print(f"  shape: {item.shape}")
                print(f"  dtype: {item.dtype}")
                print_attrs(full_name, item)
                # Preview first few elements
                try:
                    preview = item[:5]
                    print(f"  preview: {preview}")
                except Exception as e:
                    print(f"  could not preview: {e}")

    recurse(root)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_zarr.py /path/to/file.zarr")
    else:
        inspect_zarr(sys.argv[1])
