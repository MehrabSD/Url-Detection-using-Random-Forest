// Read a URL from a QR code ("quishing": phishing links printed as QR codes on
// posters, parcel notes and emails, where no link preview can warn anyone).
//
// The image never leaves the browser. It is decoded here by jsQR (vendor/, Apache
// 2.0), which is loaded only the first time it is needed, and only the decoded
// text is sent to the server for scoring, like a typed URL.

import { $ } from './util.js';

const MAX_SIDE = 1400;          // larger photos are scaled down before decoding
let decoderLoading = null;

function loadDecoder() {
  if (window.jsQR) return Promise.resolve(window.jsQR);
  decoderLoading ??= new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = 'vendor/jsQR.js';
    script.onload = () => (window.jsQR ? resolve(window.jsQR) : reject(new Error('QR decoder did not load.')));
    script.onerror = () => { decoderLoading = null; reject(new Error('QR decoder could not be loaded.')); };
    document.head.append(script);
  });
  return decoderLoading;
}

function pixels(source, width, height) {
  const scale = Math.min(1, MAX_SIDE / Math.max(width, height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(width * scale));
  canvas.height = Math.max(1, Math.round(height * scale));
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

async function decodeImage(blob) {
  const jsQR = await loadDecoder();
  // createImageBitmap reads the file directly, so no blob: URL is needed
  // (the page's content security policy allows images from this site only).
  const bitmap = await createImageBitmap(blob);
  const image = pixels(bitmap, bitmap.width, bitmap.height);
  bitmap.close?.();
  const code = jsQR(image.data, image.width, image.height, { inversionAttempts: 'attemptBoth' });
  return code?.data?.trim() || null;
}

export function setupQr({ onText, area }) {
  const status = $('qr-status');
  const say = (text) => { status.textContent = text; };
  const fileInput = $('qr-file');
  let stream = null;
  let frame = 0;

  async function fromBlob(blob) {
    if (!blob?.type?.startsWith('image/')) return say('That file is not an image.');
    say('Reading the QR code.');
    try {
      const text = await decodeImage(blob);
      if (!text) return say('No QR code found in that image. Try a sharper, closer photo.');
      say(`Read from a QR code: ${text}`);
      onText(text);
    } catch (problem) {
      say(problem.message || 'That image could not be read.');
    }
  }

  $('qr-file-button').addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) fromBlob(fileInput.files[0]);
    fileInput.value = '';
  });

  // paste an image (a screenshot of an email, say) anywhere on the page
  document.addEventListener('paste', (event) => {
    const item = [...(event.clipboardData?.items ?? [])].find((i) => i.type.startsWith('image/'));
    if (item) { event.preventDefault(); fromBlob(item.getAsFile()); }
  });
  area.addEventListener('dragover', (event) => {
    if ([...event.dataTransfer.items].some((i) => i.kind === 'file')) {
      event.preventDefault();
      area.classList.add('dropping');
    }
  });
  area.addEventListener('dragleave', (event) => {
    if (!area.contains(event.relatedTarget)) area.classList.remove('dropping');
  });
  area.addEventListener('drop', (event) => {
    area.classList.remove('dropping');
    const file = [...event.dataTransfer.files].find((f) => f.type.startsWith('image/'));
    if (file) { event.preventDefault(); fromBlob(file); }
  });

  // live camera: decode a few frames a second until a code is found
  const video = $('qr-video');
  function stop() {
    cancelAnimationFrame(frame);
    stream?.getTracks().forEach((t) => t.stop());
    stream = null;
    $('qr-camera').hidden = true;
  }
  $('qr-stop').addEventListener('click', () => { stop(); say(''); });
  $('qr-camera-button').addEventListener('click', async () => {
    if (!navigator.mediaDevices?.getUserMedia) return say('This browser cannot use a camera here. Use a photo instead.');
    try {
      const jsQR = await loadDecoder();
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
      video.srcObject = stream;
      await video.play();
      $('qr-camera').hidden = false;
      say('Hold the QR code up to the camera.');
      let last = 0;
      const scan = (time) => {
        if (!stream) return;
        if (time - last > 180 && video.videoWidth) {
          last = time;
          const image = pixels(video, video.videoWidth, video.videoHeight);
          const code = jsQR(image.data, image.width, image.height, { inversionAttempts: 'dontInvert' });
          if (code?.data) {
            stop();
            say(`Read from a QR code: ${code.data.trim()}`);
            return onText(code.data.trim());
          }
        }
        frame = requestAnimationFrame(scan);
      };
      frame = requestAnimationFrame(scan);
    } catch (problem) {
      stop();
      say(problem.name === 'NotAllowedError' ? 'Camera access was refused. Use a photo instead.'
        : 'No camera could be opened. Use a photo instead.');
    }
  });
}
