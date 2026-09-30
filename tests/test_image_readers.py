from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import zlib

import numpy as np
from PIL import Image
import tifffile

from astra.data.image_readers import open_rgb_image, TiffRgbReader, registered_transform, MAX_DECODE_BYTES
from astra.data.native_image import read_rgb_region, project_points


class ImageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        yy, xx = np.indices((35,39))
        self.rgb = np.stack((xx*3+80,yy*3+70,xx+yy+60),-1).astype(np.uint8)

    def region(self, reader, x=3,y=2,w=17,h=19):
        return read_rgb_region(reader,x_start=x,y_start=y,width=w,height=h)

    def test_lossless_tile_strip_deflate_and_png(self):
        for layout in ('tile','strip'):
            for compression in (None,'deflate'):
                with self.subTest(layout=layout,compression=compression):
                    path = self.root/'fixture.tif'
                    options = {'tile':(16,16)} if layout=='tile' else {'rowsperstrip':7}
                    tifffile.imwrite(path,self.rgb,photometric='rgb',compression=compression,**options)
                    with open_rgb_image(path) as reader:
                        np.testing.assert_array_equal(self.region(reader),self.rgb[2:21,3:20])
                        np.testing.assert_array_equal(self.region(reader,0,0,39,35),self.rgb)
                        self.assertEqual(reader.direct,compression is None)
        path = self.root/'fixture.png'; Image.fromarray(self.rgb).save(path)
        with open_rgb_image(path) as reader:
            np.testing.assert_array_equal(self.region(reader),self.rgb[2:21,3:20])

    def test_jpeg_matches_pillow_decoding_but_is_lossy(self):
        path = self.root/'fixture.jpg'; Image.fromarray(self.rgb).save(path,quality=95,subsampling=0)
        with Image.open(path) as image:
            reference = np.array(image)
        with open_rgb_image(path) as reader:
            np.testing.assert_array_equal(self.region(reader),reference[2:21,3:20])
        error = np.abs(reference.astype(float)-self.rgb)
        self.assertGreater(error.max(),0)
        self.assertLess(error.max(),10)  # fixture behavior only, not a universal JPEG error bound

    def test_boundary_padding_preserves_coordinates(self):
        path = self.root/'fixture.png'; Image.fromarray(self.rgb).save(path)
        with open_rgb_image(path) as reader:
            region = self.region(reader,-2,-1,6,5)
        np.testing.assert_array_equal(region[1:,2:],self.rgb[:4,:4])
        self.assertTrue((region[:1] == 255).all())
        self.assertTrue((region[:,:2] == 255).all())

    def test_tiff_and_exif_orientation_rejected(self):
        path = self.root/'oriented.tif'
        tifffile.imwrite(path,self.rgb,photometric='rgb',extratags=[(274,'H',1,6,False)])
        with self.assertRaisesRegex(ValueError,'orientation=1'):
            open_rgb_image(path)
        path = self.root/'oriented.jpg'
        exif = Image.Exif(); exif[274] = 6
        Image.fromarray(self.rgb).save(path,exif=exif)
        with self.assertRaisesRegex(ValueError,'orientation=1'):
            open_rgb_image(path)

    def test_dtype_channels_alpha_palette_and_icc_rejected(self):
        for array, options in [(self.rgb.astype(np.uint16),{'photometric':'rgb'}),
                               (self.rgb[:,:,0],{}),
                               (np.dstack((self.rgb,np.full(self.rgb.shape[:2],255,np.uint8))),{'photometric':'rgb'}),
                               (self.rgb.transpose(2,0,1),{'photometric':'rgb','planarconfig':'separate'})]:
            path = self.root/'bad.tif'; tifffile.imwrite(path,array,**options)
            with self.subTest(shape=array.shape,dtype=array.dtype), self.assertRaises(ValueError):
                open_rgb_image(path)
        for mode in ('RGBA','L','P'):
            path = self.root/'bad.png'; Image.new(mode,(8,8)).save(path)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                open_rgb_image(path)
        path = self.root/'icc.png'; Image.fromarray(self.rgb).save(path,icc_profile=b'unsupported-profile')
        with self.assertRaisesRegex(ValueError,'ICC'):
            open_rgb_image(path)

    def test_png_16bit_truecolor_rejected_before_pillow_downconversion(self):
        # Minimal valid 16-bit RGB PNG, generated from the PNG chunk specification.
        def chunk(name,data):
            return struct.pack('>I',len(data))+name+data+struct.pack('>I',zlib.crc32(name+data))
        path = self.root/'16bit.png'
        path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',1,1,16,2,0,0,0))+
                         chunk(b'IDAT',zlib.compress(b'\0\0\1\0\2\0\3'))+chunk(b'IEND',b''))
        with self.assertRaisesRegex(ValueError,'8-bit truecolor'):
            open_rgb_image(path)

    def test_missing_calibration_and_explicit_transform(self):
        with self.assertRaisesRegex(ValueError,'missing physical calibration'):
            registered_transform({'width':35,'dpi':300})
        matrix = registered_transform({'spot_to_image':[[4,0,1.5],[0,4,1.5],[0,0,1]]})
        x,y = project_points(matrix,np.array([0,3]),np.array([1,4]))
        np.testing.assert_array_equal(x,[1.5,13.5]); np.testing.assert_array_equal(y,[5.5,17.5])
        with self.assertRaisesRegex(ValueError,'invertible'):
            registered_transform({'spot_to_image':np.zeros((3,3))})

    def test_large_tiff_reads_requested_tiles_with_bounded_cache(self):
        path = self.root/'large.tif'
        tile = np.full((256,256,3),113,np.uint8)
        # 768 MiB decoded slide, tiny compressed fixture; no large RGB allocation.
        tifffile.imwrite(path,(tile for _ in range(64*64)),shape=(16384,16384,3),dtype=np.uint8,
                         tile=(256,256),photometric='rgb',compression='deflate')
        with mock.patch.object(tifffile.TiffPage,'asarray',side_effect=AssertionError('whole-slide decode forbidden')):
            with TiffRgbReader(path,max_cache_bytes=tile.nbytes) as reader:
                self.assertEqual(reader.decoded_segments,0)
                np.testing.assert_array_equal(self.region(reader,3,4,5,6),np.full((6,5,3),113,np.uint8))
                self.assertEqual(reader.decoded_segments,1)
                self.region(reader,270,4,5,6)
                self.assertEqual(reader.decoded_segments,2)
                self.assertLessEqual(reader._cache_bytes,tile.nbytes)
                self.assertEqual(len(reader._cache),1)

    def test_large_raster_and_single_strip_rejected_before_decode(self):
        path = self.root/'large.tif'
        # Header alone suffices to reject oversized strip geometry.
        tifffile.imwrite(path,data=None,shape=(5000,5000,3),dtype=np.uint8,photometric='rgb')
        with mock.patch.object(tifffile.TiffPage,'asarray',side_effect=AssertionError('decode forbidden')):
            with self.assertRaisesRegex(ValueError,'segment exceeds'):
                open_rgb_image(path)
        fake = mock.MagicMock()
        fake.format, fake.size = 'PNG',(5000,5000)
        path = self.root/'large.png'; path.write_bytes(b'\x89PNG')
        with mock.patch('astra.data.image_readers.Image.open') as opening:
            opening.return_value.__enter__.return_value = fake
            with self.assertRaisesRegex(ValueError,'exceeds 64 MiB'):
                open_rgb_image(path)
        fake.load.assert_not_called()

    def test_missing_codec_has_actionable_error_and_no_fallback(self):
        path = self.root/'deflate.tif'; tifffile.imwrite(path,self.rgb,photometric='rgb',compression='deflate',tile=(16,16))
        with TiffRgbReader(path) as reader:
            reader.decode = mock.Mock(side_effect=ValueError('requires imagecodecs'))
            with self.assertRaisesRegex(ValueError,'imagecodecs.*no full-image fallback'):
                self.region(reader)

    def test_missing_tifffile_and_multipage_rejected(self):
        path = self.root/'fixture.tif'; tifffile.imwrite(path,self.rgb,photometric='rgb')
        with mock.patch.dict('sys.modules',{'tifffile':None}):
            with self.assertRaisesRegex(ImportError,'section extra'):
                open_rgb_image(path)
        with tifffile.TiffWriter(path) as writer:
            writer.write(self.rgb,photometric='rgb'); writer.write(self.rgb,photometric='rgb')
        with self.assertRaisesRegex(ValueError,'multipage'):
            open_rgb_image(path)


if __name__ == '__main__':
    unittest.main()
