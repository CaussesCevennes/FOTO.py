# FOTO.py
Fourier Transform Textural Ordination in Python

Usage

```
from FOTO import FOTO

if __name__ == '__main__': #guard needed to make the client multiprocess safe

  foto = FOTO(inPath, blockSize=64, method='BLOCK', maxSample=29)
  foto.run()
  foto.writeRGB(outPath)
  foto.plotPCA()
```
