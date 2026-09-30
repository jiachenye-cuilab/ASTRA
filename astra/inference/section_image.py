import inspect
from pathlib import Path
from astra.data import native_image as image


def flat_integrator():
    source = inspect.getsource(image.aggregate_density_region)
    before = '        for channel_index, channel in enumerate(DENSITY_CHANNELS):'
    after = '''        width = source_x1 - source_x0
        i00 = local_y0 * width + local_x0
        i01 = local_y0 * width + local_x1
        i10 = local_y1 * width + local_x0
        i11 = local_y1 * width + local_x1
        for channel_index, channel in enumerate(DENSITY_CHANNELS):'''
    assert source.count(before) == 1
    source = source.replace(before, after)
    before = '''            source = source_density[channel]
            sampled = (
                source[local_y0, local_x0] * w00
                + source[local_y0, local_x1] * w01
                + source[local_y1, local_x0] * w10
                + source[local_y1, local_x1] * w11
            )'''
    after = '''            source = source_density[channel].reshape(-1)
            sampled = source[i00] * w00
            sampled += source[i01] * w01
            sampled += source[i10] * w10
            sampled += source[i11] * w11'''
    assert source.count(before) == 1
    source = source.replace(before, after)
    namespace = dict(image.aggregate_density_region.__globals__)
    exec(compile(source, str(Path(__file__).with_name('flat_integrator.in_memory.py')), 'exec'), namespace)
    return namespace['aggregate_density_region']


aggregate_density_region = flat_integrator()
