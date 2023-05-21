from random import random


def writeBmp(filename, width, height, colorTables, pixels):
    with open(filename, 'wb') as f:
        lenOfColors = len(colorTables)
        numOfColors = lenOfColors >> 2
        lenOfPixels = len(pixels)
        bfOffBits = 14 + 0x28 + lenOfColors
        fileSize = bfOffBits + lenOfPixels

        # FILE_HEADER
        b = bytearray([0x42, 0x4d])  # シグネチャ 'BM'
        b.extend(fileSize.to_bytes(4, 'little'))  # ファイルサイズ
        b.extend((0).to_bytes(2, 'little'))  # 予約領域
        b.extend((0).to_bytes(2, 'little'))  # 予約領域
        # ファイル先頭から画像データまでのオフセット[byte] ※誤った値だとアプリによっては表示失敗した
        b.extend(bfOffBits.to_bytes(4, 'little'))

        # INFO_HEADER
        b.extend((0x28).to_bytes(4, 'little'))  # ヘッダーサイズ
        b.extend(width.to_bytes(4, 'little'))  # 幅[dot]
        b.extend(height.to_bytes(4, 'little'))  # 高さ[dot]
        b.extend((1).to_bytes(2, 'little'))  # プレーン数 常に1
        b.extend((8).to_bytes(2, 'little'))  # byte/1pixel(1byteを表すために必要なbit)
        b.extend((0).to_bytes(4, 'little'))  # 圧縮形式 0 - BI_RGB（無圧縮）
        b.extend(lenOfPixels.to_bytes(4, 'little'))  # 画像データサイズ[byte]
        b.extend((0).to_bytes(4, 'little'))  # X方向解像度[dot/m] 0の場合もある
        b.extend((0).to_bytes(4, 'little'))  # Y方向解像度[dot/m] 0の場合もある
        # 使用する色の数 ※0だとアプリによっては表示失敗した
        b.extend(numOfColors.to_bytes(4, 'little'))
        b.extend((0).to_bytes(4, 'little'))  # 重要な色の数 0の場合もある

        # COLOR_TABLES
        b.extend(colorTables)

        # DATA
        b.extend(pixels)

        f.write(b)


def read256BmpFile(filename):
    with open(filename, 'rb') as f:
        # いわゆる '256bmp' 専用。それ以外での動作は未定義。

        # BMP file header
        # bfType         = f.read(2) # 今後エラーチェック追加時に利用したいときなど用。以降の類似したコメントアウトも同様
        f.read(2)
        f.read(4)  # bfSize         = int.from_bytes(f.read(4), byteorder='little')
        f.read(2)  # bfReserved1    = int.from_bytes(f.read(2), byteorder='little')
        f.read(2)  # bfReserved2    = int.from_bytes(f.read(2), byteorder='little')
        f.read(4)  # bfOffBits      = int.from_bytes(f.read(4), byteorder='little')

        # BMP information header
        f.read(4)  # bcSize         = int.from_bytes(f.read(4), byteorder='little')
        bcWidth = int.from_bytes(f.read(4), byteorder='little')
        bcHeight = int.from_bytes(f.read(4), byteorder='little')
        f.read(2)  # bcPlanes       = int.from_bytes(f.read(2), byteorder='little')
        f.read(2)  # bcBitCount     = int.from_bytes(f.read(2), byteorder='little')
        f.read(4)  # biCompression  = int.from_bytes(f.read(4), byteorder='little')
        f.read(4)  # biSizeImage    = int.from_bytes(f.read(4), byteorder='little')
        f.read(4)  # biXPixPerMeter = int.from_bytes(f.read(4), byteorder='little')
        f.read(4)  # biYPixPerMeter = int.from_bytes(f.read(4), byteorder='little')
        biClrUsed = int.from_bytes(f.read(4), byteorder='little')
        f.read(4)  # biCirImportant = int.from_bytes(f.read(4), byteorder='little')

        if not biClrUsed:
            biClrUsed = 256

        # color table
        colorTable = f.read(biClrUsed * 4)

        # pixels
        pixels = f.read()

        return (bcWidth, bcHeight, colorTable, pixels)


# Blue, Green, Red, Reserved, ...
colorTables = [0x00, 0x00, 0xFF, 0x00,  0x00, 0xFF, 0x00, 0x00]
pixels = [0x00, 0x00, 0x01, 0x01,  0x00, 0x01, 0x00, 0x01]
writeBmp('sample.bmp', 4, 2, colorTables, pixels)


(width, height, colorTable, pixels) = read256BmpFile('sample.bmp')
# print(pixels[2])
