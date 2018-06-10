import os
import platform
import shutil
import tempfile
import zipfile

try:
    from urllib.request import urlopen  # NOQA
except ImportError:
    from urllib import urlopen

BASE_URL = "https://github.com/maciejkula/sbr-sys/releases/download/"
LINUX_URL = "untagged-8b9d185393b92ca20ccb/libsbr_linux.zip"
DARWIN_URL = "untagged-8438cacd506366a30457/libsbr_darwin.zip"


def download_release(platform):

    if platform == "Linux":
        url = BASE_URL + LINUX_URL
        prefix = "linux"
    else:
        url = BASE_URL + DARWIN_URL
        prefix = "darwin"

    try:
        os.makedirs("lib")
    except FileExistsError:
        pass

    workdir = tempfile.mkdtemp()
    destpath = os.path.join(workdir, "libsbr.zip")

    with open(destpath, "wb") as destfile:
        response = urlopen(url)
        destfile.write(response.read())

    with zipfile.ZipFile(destpath, mode="r") as zfile:
        zfile.extract("{}/sse/libsbr_sys.a".format(prefix), workdir)

    shutil.copy(
        os.path.join(workdir, prefix, "sse", "libsbr_sys.a"), os.path.join("lib", "libsbr_sys.a")
    )


if __name__ == "__main__":

    download_release(platform.system())
