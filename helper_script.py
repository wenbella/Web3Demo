from pathlib import Path
import json
import ipfsapi
import os

metadata_template = {
    "name": "",
    "description": "",
    "image": "",
}


def create_metadata(token_id, prompt, image_dir):
    abs_path = os.path.dirname(os.path.realpath(__file__)) + "/metadata/"
    metadata_file_name = (
        abs_path + f"{token_id}.json"
    )
    if not os.path.exists(abs_path):
        os.mkdir(abs_path)
    collectible_metadata = metadata_template
    if Path(metadata_file_name).exists():
        print(f"{metadata_file_name} already exists! Delete it to overwrite")
    else:
        print(f"Creating Metadata file: {metadata_file_name}")
        collectible_metadata["name"] = "Stable Diffusion" + " " + token_id
        collectible_metadata["description"] = prompt
        # 连接IPFS，需要先启动节点服务器daemon
        api = ipfsapi.connect('127.0.0.1', 5001)
        # upload image to ipfs
        image_uri = upload_img_to_ipfs(api, image_dir)
        collectible_metadata["image"] = image_uri
        with open(metadata_file_name, "w") as file:
            json.dump(collectible_metadata, file)
        # upload metadata to ipfs
        ret = upload_metadata_to_ipfs(api, metadata_file_name)
        print("metadata url is {}".format(ret))


# curl -X POST -F file=@metadata/goerli/0-SHIBA_INU.json http://localhost:5001/api/v0/add

def upload_img_to_ipfs(api, image_dir):
    # 上传文件
    res = api.add(image_dir)
    ipfs_hash = res["Hash"]
    image_uri = f"https://ipfs.io/ipfs/{ipfs_hash}"
    print(image_uri)
    return image_uri


def upload_metadata_to_ipfs(api, filepath):
    # 上传文件
    res = api.add(filepath)
    ipfs_hash = res["Hash"]
    metadata_uri = f"https://ipfs.io/ipfs/{ipfs_hash}"
    print(metadata_uri)
    return metadata_uri


if __name__ == "__main__":
    # with Path("/Users/bella/Desktop/nft/nft-demo/img/pug.png").open("rb") as fp:
    #     image_binary = fp.read()
    create_metadata("000000000000000001", "test prompt", "/Users/bella/Desktop/nft/nft-demo/img/pug.png")
