# FOTO.py
Fourier Transform Textural Ordination in Python

Usage

```
if __name__ == '__main__': #guard needed to make the client mutiprocess safe
  foto = FOTO(inPath, blockSize=64, method='BLOCK', maxSample=29)
  foto.run()
  foto.writeRGB(outPath)
  foto.plotPCA()
```
