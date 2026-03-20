import cv2   
import numpy as np
import math
from zernike import RZern
import globals
import matplotlib.pyplot as plt

# Whether to use unet for recognition
if globals.USE_UNET_MASK:
    from unet import Unet
    from PIL import Image
    try: unet = Unet()
    except:
        print('unet loading failed')
        globals.USE_UNET_MASK = False

class dhmPic:
    def __init__(self, rawdata):
        self.rawdata = rawdata
        self.m, self.n = rawdata.shape
        self.angle = None
        self.phase = None
        self.FMZF_phase = None
        self.PMZF_phase = None
        self.unet_mask = None
        # Perform Fourier transform on the initial image
        if rawdata is not None:
            dft = cv2.dft(np.float32(self.rawdata),flags=cv2.DFT_COMPLEX_OUTPUT)
            self.dftshift = np.fft.fftshift(dft)

    # Unwrapping function
    def dhm_unwrap(self, ph_wrap):
        # this method is based on Least Square Estimate
        # step 1: construct periodic psai
        M, N = ph_wrap.shape
        ph1 = np.zeros((M+2, N+2))
        ph1[1:M+1, 1:N+1] = ph_wrap

        ph1[0, :] = ph1[2, :]
        ph1[M+1, :] = ph1[M, :]
        ph1[1:M+1, 0] = ph1[1:M+1, 2]
        ph1[1:M+1, N+1] = ph1[1:M+1, N]

        # step 2: construct delta_v and delta_h
        delta_v = np.zeros((M+1, N))
        for m in range(M+1):
            for k in range(N):
                delta_v[m, k] = ph1[m+1, k+1] - ph1[m, k+1]
                if delta_v[m, k] <= -math.pi:
                    delta_v[m, k] += 2*math.pi
                if delta_v[m, k] > math.pi:
                    delta_v[m, k] -= 2*math.pi

        delta_h = np.zeros((M, N+1))
        for n in range(N+1):
            for k in range(M):
                delta_h[k, n] = ph1[k+1, n+1] - ph1[k+1, n]
                if delta_h[k, n] <= -math.pi:
                    delta_h[k, n] += 2*math.pi
                if delta_h[k, n] > math.pi:
                    delta_h[k, n] -= 2*math.pi

        # step 3: construct rou
        rou = np.zeros((M, N))
        for m in range(M):
            for n in range(N):
                rou[m, n] = delta_v[m+1, n] - delta_v[m, n] + delta_h[m, n+1] - delta_h[m, n]

        # step 4: FFT
        rou1 = np.zeros((M, 2*N-2))
        rou1[:M, :N] = rou
        for m in range(M):
            for n in range(N, 2*N-2):
                rou1[m, n] = rou[m, 2*N-2-n]

        sp = np.fft.fft(rou1, axis=1)
        sp1 = np.real(sp[:M, :N])

        sp2 = np.zeros((2*M-2, N))
        sp2[:M, :N] = sp1
        for n in range(N):
            for m in range(M, 2*M-2):
                sp2[m, n] = sp1[2*M-2-m, n]

        sp = np.fft.fft(sp2, axis=0)
        sp1 = np.real(sp[:M, :N])

        # step 5: construct capital psai
        psai = np.zeros((M, N))
        for m in range(M):
            for n in range(N):
                if m == 0 and n == 0:
                    psai[m, n] = 0
                else:
                    psai[m, n] = sp1[m, n] / (2*math.cos(math.pi*m/(M-1)) + 2*math.cos(math.pi*n/(N-1)) - 4)

        # step 6: inverse FFT according to step2
        rou1 = np.zeros((M, 2*N-2))
        rou1[:M, :N] = psai
        for m in range(M):
            for n in range(N, 2*N-2):
                rou1[m, n] = psai[m, 2*N-2-n]

        sp = np.fft.ifft(rou1, axis=1)
        sp1 = np.real(sp[:M, :N])

        sp2 = np.zeros((2*M-2, N))
        sp2[:M, :N] = sp1
        for n in range(N):
            for m in range(M, 2*M-2):
                sp2[m, n] = sp1[2*M-2-m, n]

        sp = np.fft.ifft(sp2, axis=0)
        sp1 = np.real(sp[:M, :N])

        phase = sp1

        return phase

    # Identify the center point of the binary image
    def find_left_center(self):
        binary_img = self.img_thresh
        # Edge detection
        contours= cv2.findContours(binary_img, cv2.RETR_TREE, 
                                cv2.CHAIN_APPROX_SIMPLE)[0]
        cX = None
        cY = None
        for i in contours:
            M = cv2.moments(i)
            if M["m00"] :
                x = int(M["m10"] / M["m00"])
                y = int(M["m01"] / M["m00"])
                if cX is None or x < cX:
                    cX = x
                    cY = y
        return cX, cY

    # Window filter position recognition
    def filter_init(self, thresh=140):
        dftshift = self.dftshift
        res1 = np.log(cv2.magnitude(dftshift[:,:,0], dftshift[:,:,1]))
        res1_gray = cv2.convertScaleAbs(res1, alpha=(255.0/np.max(res1)))
        self.res1_gray = res1_gray
        # Take the left half
        left_res1 = res1_gray[:, :(res1.shape[1] // 2)]
        # Gaussian filtering
        blurred = cv2.GaussianBlur(left_res1,(113,113),0)
        # Binarization
        self.img_thresh = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)[1]
        # Automatically recognize filter window position
        self.filter_cx, self.filter_cy = self.find_left_center()
        return self.filter_cx, self.filter_cy
    
    # Window filtering and phase restoration
    def filt_and_ifft(self, scale = 5, radi=50):
        
        filt_size=[self.m//scale, self.n//scale]
        
        if filt_size[0]<2*radi+1 or filt_size[1]<2*radi+1:
            # If the preset size is smaller than the window filter size, set to original image size
            filt_size = [self.m, self.n]
        
        cx = self.filter_cx
        cy = self.filter_cy
        r = radi
        shift = self.dftshift

        # Filter initialization
        filtered = np.zeros((filt_size[0],filt_size[1],2))
        # Window filtering
        filtered[filtered.shape[0]//2-r:filtered.shape[0]//2+r, 
                filtered.shape[1]//2-r:filtered.shape[1]//2+r
                ] = shift[cy-r:cy+r, cx-r:cx+r]
        # Perform inverse Fourier transform on filtered
        filtered_shift = np.fft.ifftshift(filtered)
        filtered_idft = cv2.idft(filtered_shift)
        # Calculate phase
        ang = filtered_idft[:,:,0] + filtered_idft[:,:,1]*1j
        ang = np.angle(ang)
        self.angle = ang
        return ang
    
    # Phase unwrapping
    def get_phase(self):
        self.phase = self.dhm_unwrap(self.angle) * -1
        return self.phase

    # Unet recognizes background area
    def unet_get_mask(self):
        if globals.USE_UNET_MASK:
            phase = cv2.normalize(self.phase, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            image = Image.fromarray(phase).convert('L')
            mask = unet.detect_image(image, count=False, name_classes=["background","object"])
            self.unet_mask = mask
            return mask
        else:
            return

    # zernike fitting
    def fitting(self, mask = None, zn=2):
        img_phase = self.phase
        if mask is not None:
            if mask.shape != img_phase.shape:
                mask = cv2.resize(mask, (img_phase.shape[1], img_phase.shape[0]))
            if mask.dtype != bool:
                mask = cv2.threshold(mask, 140, 255, cv2.THRESH_BINARY)[1]
            img_phase = img_phase.astype(float)
            img_phase[mask == 0] = np.nan
        cart = RZern(zn)
        L, K = img_phase.shape
        ddx = np.linspace(-1.0, 1.0, K)
        ddy = np.linspace(-1.0, 1.0, L)
        xv, yv = np.meshgrid(ddx, ddy)
        cart.make_cart_grid(xv, yv, unit_circle=False)
        c1 = cart.fit_cart_grid(img_phase)[0]
        # print(c1) 
        self.surf_c = c1
        Phi = cart.eval_grid(c1, matrix=True)
        self.fit_surf = Phi
        phase_fitted = self.phase - Phi
        if np.max(phase_fitted)<0 :
            phase_fitted = -phase_fitted
        # FMZF and PMZF are stored separately
        if mask is None:
            self.FMZF_phase = phase_fitted
        else:
            self.PMZF_phase = phase_fitted
            self.FMZF_phase = self.fitting()
        return phase_fitted

    # Display filtering situation
    def filter_show(self, window_On = 'True'):
        thresh = self.img_thresh
        filter_cx = self.filter_cx
        filter_cy = self.filter_cy
        radi = 50
        # Draw rectangle
        if filter_cx is not None and filter_cy is not None:
            # Draw rectangle
            cv2.rectangle(thresh,(filter_cx-radi,filter_cy-radi),
                          (filter_cx+radi,filter_cy+radi),100,3)
        out_img = cv2.resize(thresh, (320, 480))
        if window_On:
            cv2.imshow('Review', out_img)
            cv2.waitKey()
            cv2.destroyAllWindows()
        return out_img

# Inverse unwrapping function
def phase_wrap(phase):
    wrapped_phase = phase - (np.pi + np.min(phase))
    while np.max(wrapped_phase) > np.pi:
        # Subtract 2pi from all elements greater than pi in the matrix
        wrapped_phase[wrapped_phase > np.pi] -= 2* np.pi
    return wrapped_phase

# Normalization
def result_remap(matrix, trans_value, scale_value):
    out_matrix = (matrix + trans_value) *scale_value
    # Rounding
    out_matrix = np.round(out_matrix)
    # Assign 0 to values less than 0 after rounding, and 255 to values greater than 255
    out_matrix = np.clip(out_matrix, 0, 255)
    out_matrix = out_matrix.astype(np.uint8)
    return out_matrix

def dhm_angle2phase(filename):
    if filename is None:
        filename = input('Enter video filename:')
    try:
        angle = np.fromfile(filename, dtype=np.uint8)
        angle = cv2.imdecode(angle, cv2.IMREAD_GRAYSCALE)  # Read as grayscale image
    except Exception as e:
        print(f'Unable to open file: {e}')
        return
    if angle is None:
        print('Unable to open file')
        return
    raw = np.ones_like(angle)  # RGB three channels
    angle = np.interp(angle, (np.min(angle), np.max(angle)), (-np.pi, np.pi))
    pic = dhmPic(raw)
    pic.angle = angle
    phase = pic.get_phase()
    phase = cv2.normalize(pic.phase, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    phase = cv2.cvtColor(phase, cv2.COLOR_GRAY2BGR)
    # Display grayscale image
    cv2.imshow('phase', phase)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ===================== Accelerated Helpers & dhm_video =====================
# Multi-process + optional GPU (CuPy) acceleration for per-frame processing.
# Keeps the same interface as before.
import multiprocessing as mp
from functools import partial

# Optional GPU backend
try:
    import cupy as cp
    _cupy_available = True
except Exception:
    cp = None
    _cupy_available = False

# Optional fast unwrap (skimage)
try:
    from skimage.restoration import unwrap_phase as _unwrap_fast
    _unwrap_fast_available = True
except Exception:
    _unwrap_fast_available = False

def _unwrap_phase_np(angle_np):
    """
    Fast phase unwrapping with skimage if available; otherwise
    falls back to original least-squares unwrapping method.
    """
    if _unwrap_fast_available:
        return _unwrap_fast(angle_np).astype(angle_np.dtype) * -1  # preserve previous sign
    # Fall back to the original method in this module
    pic_stub = dhmPic(np.zeros_like(angle_np, dtype=np.float32))
    pic_stub.angle = angle_np
    return pic_stub.dhm_unwrap(angle_np) * -1

def _compute_angle_fft(frame_gray, cx, cy, radi, scale=5, use_gpu=False):
    """
    Compute angle via frequency windowing + IFFT.
    Uses CuPy GPU if available and use_gpu=True, else NumPy.
    """
    xp = cp if (use_gpu and _cupy_available) else np
    # convert to array on target device
    arr = xp.asarray(frame_gray, dtype=xp.float32)
    M, N = arr.shape
    # FFT and shift
    dft = xp.fft.fft2(arr)
    dftshift = xp.fft.fftshift(dft)

    # determine filtered spectrum size
    filt_h = M // scale
    filt_w = N // scale
    if filt_h < 2 * radi + 1 or filt_w < 2 * radi + 1:
        filt_h, filt_w = M, N

    filtered = xp.zeros((filt_h, filt_w), dtype=xp.complex64)
    h2, w2 = filt_h // 2, filt_w // 2
    r = radi

    # crop from the original spectrum and paste to the center
    patch = dftshift[cy - r: cy + r, cx - r: cx + r]
    filtered[h2 - r:h2 + r, w2 - r:w2 + r] = patch

    # IFFT back to spatial domain
    filtered_shift = xp.fft.ifftshift(filtered)
    spatial = xp.fft.ifft2(filtered_shift)

    # angle
    angle = xp.angle(spatial).astype(xp.float32)

    # bring back to CPU (numpy)
    return cp.asnumpy(angle) if xp is cp else angle

def _zernike_fit(phase_np, mask_np=None, zn=2):
    """
    Zernike fitting with optional mask (numpy arrays).
    Mirrors logic in dhmPic.fitting, returns (FMZF_phase, PMZF_phase)
    depending on whether mask is None or not.
    """
    # Prepare image and mask
    img_phase = phase_np.astype(float).copy()
    L, K = img_phase.shape

    if mask_np is not None:
        if mask_np.shape != img_phase.shape:
            mask_np = cv2.resize(mask_np, (K, L))
        if mask_np.dtype != bool:
            mask_np = cv2.threshold(mask_np, 140, 255, cv2.THRESH_BINARY)[1]
        img_phase[mask_np == 0] = np.nan

    cart = RZern(zn)
    ddx = np.linspace(-1.0, 1.0, K)
    ddy = np.linspace(-1.0, 1.0, L)
    xv, yv = np.meshgrid(ddx, ddy)
    cart.make_cart_grid(xv, yv, unit_circle=False)
    c1 = cart.fit_cart_grid(img_phase)[0]
    Phi = cart.eval_grid(c1, matrix=True)
    phase_fitted = phase_np - Phi

    if np.max(phase_fitted) < 0:
        phase_fitted = -phase_fitted

    return phase_fitted

def _maybe_unet_mask(phase_np):
    """
    Create mask using existing Unet (if enabled).
    """
    if not getattr(globals, "USE_UNET_MASK", False):
        return None
    try:
        # reuse loaded "unet" from module scope if available
        phase_u8 = cv2.normalize(phase_np, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        image = Image.fromarray(phase_u8).convert('L')
        mask = unet.detect_image(image, count=False, name_classes=["background","object"])
        return mask
    except Exception as e:
        print("UNet mask failed:", e)
        return None

def ensure_bgr(frame, w, h):
    """Ensure output is uint8 (h,w,3) BGR"""
    if frame is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if len(frame.shape) == 2:  # Grayscale image
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 3:  # 3 channels
        if frame.shape[0] != h or frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h))
    else:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
    return frame

if __name__ == '__main__':
    # Load wrapped phase from sample/phase_wrap.csv
    phase_wrap = np.loadtxt('sample/phase_wrap.csv', delimiter=',')
    
    pic = dhmPic()
    pic.angle = phase_wrap
    # Unwrap the phase
    phase = pic.dhm_unwrap(phase_wrap)
    
    # Save unwrapped phase to phase.csv
    np.savetxt('sample/phase.csv', phase, delimiter=',')
    
    # Plot and save phase.png with 'viridis' colormap
    plt.imshow(phase, cmap='viridis')
    plt.axis('off')
    plt.savefig('sample/phase.png', bbox_inches='tight')
    plt.close()
    
    # Create dhmPic object
    pic = dhmPic(None)
    pic.angle = phase
    
    # Get FMZF phase
    fmzf_phase = pic.fitting()
    np.savetxt('sample/FMZF_phase.csv', fmzf_phase, delimiter=',')
    plt.imshow(fmzf_phase, cmap='viridis')
    plt.axis('off')
    plt.savefig('sample/FMZF_phase.png', bbox_inches='tight')
    plt.close()
    
    # Get PMZF phase
    mask = pic.unet_get_mask()
    pmzf_phase = pic.fitting(mask=mask)
    np.savetxt('sample/PMZF_phase.csv', pmzf_phase, delimiter=',')
    plt.imshow(pmzf_phase, cmap='viridis')
    plt.axis('off')
    plt.savefig('PMZF_phase.png', bbox_inches='tight')
    plt.close()