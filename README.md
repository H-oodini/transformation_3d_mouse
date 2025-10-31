# 3D Transformation with OpenCV, PiCam2 Chessboard Calibration & NAS Sync

This repository documents a workflow for performing a 3D transformation using OpenCV with images captured from a Raspberry Pi Camera Module 2 (PiCam2).  
The Pi captures chessboard calibration images, which are then used to compute intrinsic camera parameters required for 3D reconstruction.  
Image acquisition is handled directly on the Raspberry Pi using `libcamera-still`, and all captured calibration images are automatically synchronized to a TrueNAS server via SMB using `rsync`.

## Features

* Capture calibration images using libcamera-still
* Automatic timestamped output directories
* Metadata manifest generation
* Automatic transfer of the entire captures directory to TrueNAS via SMB
* Optional removal of images from the Pi after syncing
* Persistent SMB mount configured via /etc/fstab

## Raspberry Pi Setup
1. Capture Directory
Images are saved locally under:

```
~/Desktop/captures/
    chessboard_left_<timestamp>_<resolution>/
        chessboard_01.jpg
        chessboard_02.jpg
        ...
        manifest.txt
```

This folder is later synced to the NAS.

## SMB Mount Configuration (Raspberry Pi)

1. Mountpoint
Create a mount directory:
`sudo mkdir -p /mnt/truenas`

2. Credentials File
Store TrueNAS SMB credentials in:
`/etc/smbcredentials-truenas`


Format:
```
username=YOUR_USERNAME
password=YOUR_PASSWORD
```


Protect the file:
`sudo chmod 600 /etc/smbcredentials-truenas`

3. /etc/fstab Entry

To ensure the NAS share is mounted automatically at boot:
```
# TrueNAS SMB Share
//10.2.2.101/HUB-DATA /mnt/truenas cifs \
  credentials=/etc/smbcredentials-truenas,vers=3.1.1,sec=ntlmssp,domain=WORKGROUP,uid=1000,gid=1000,file_mode=0664,dir_mode=0775 0 0
``` 

Apply the mount:
`sudo mount -a`

Verify:
`ls /mnt/truenas`

## capture.py
Image Capture

The script uses:
`libcamera-still -o <file> --width <w> --height <h> --nopreview -t 1`

Images are saved into a timestamped subfolder inside ~/Desktop/captures.

## Syncing to NAS

After capture, the script synchronizes the entire local captures directory to:
`/mnt/truenas/FOR2591/jazmin/3D_reconstruction/`

## Running the Script

Run directly on the Raspberry Pi:
`python3 capture.py`

The script will:
1. Capture the requested number of images
2. Store them in ~/Desktop/captures/...
3. Write a manifest.txt file
4. Sync the folder to the NAS
5. clean up local data

## Requirements
* Raspberry Pi OS (Bullseye or newer)
* Python 3
* libcamera (libcamera-still)
* rsync
* cifs-utils

# Transformation from 2D images to 3D of a mouse

- transformation.py: the script does the 3D transformation. It uses yolo model and SIFT for the feature dectecion needed for the 3D reconstruction.
             see the pages to see how this work: https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html and http://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
             but before using this the frames should be already selected for example with frame_selection.py script



