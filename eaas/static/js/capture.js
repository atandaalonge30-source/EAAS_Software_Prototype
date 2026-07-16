async function initCamera(videoEl) {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 480 }, audio: false });
    videoEl.srcObject = stream;
    await videoEl.play();
    return stream;
  } catch (err) {
    document.getElementById('scan-status').innerText = 'Camera unavailable: ' + err.message;
    return null;
  }
}

function grabFrame(videoEl) {
  const canvas = document.createElement('canvas');
  canvas.width = 480; canvas.height = 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.9);
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
