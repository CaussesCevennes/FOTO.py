# FOTO.py
Fourier Transform Textural Ordination in Python

**Deprecration warning :** This project is no longer maintained, please use the more powerfull [fototex](https://pypi.org/project/fototex/) package.


Usage

```python
from FOTO import FOTO

if __name__ == '__main__': #guard needed to make the client multiprocess safe

  foto = FOTO(inPath, blockSize=64, method='BLOCK', maxSample=29, normalize=True)
  foto.run()
  foto.writeRGB(outPath)
  foto.plotPCA()
```
