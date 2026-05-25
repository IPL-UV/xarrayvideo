import json
import logging
import time
import shutil
import subprocess
import tempfile
from pathlib import Path

import ffmpeg
import numpy as np


logger = logging.getLogger(__name__)


EXTERNAL_VIDEO_CODECS = {
    'vvenc': {
        'extension': '.266',
        'encoder_bins': ('vvencapp', 'vvencFFapp'),
        'decoder_bins': ('vvdecapp',),
    },
    'uavs3e': {
        'extension': '.avs3',
        'encoder_bins': ('uavs3enc', 'uavs3e'),
        'decoder_bins': ('uavs3dec', 'uavs3d'),
    },
    'vtm': {
        'extension': '.vvc',
        'encoder_bins': ('EncoderAppStatic', 'EncoderApp'),
        'decoder_bins': ('DecoderAppStatic', 'DecoderApp'),
    },
}


def is_external_video_codec(codec_name):
    return codec_name in EXTERNAL_VIDEO_CODECS


def get_external_video_extension(codec_name):
    return EXTERNAL_VIDEO_CODECS[codec_name]['extension']


def _find_binary(candidates):
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _metadata_path(video_path):
    return Path(video_path).with_suffix(Path(video_path).suffix + '.json')


def _write_metadata(metadata_path, metadata):
    serializable = {**metadata}
    if isinstance(serializable.get('NORMALIZED'), np.bool_):
        serializable['NORMALIZED'] = bool(serializable['NORMALIZED'])
    with open(metadata_path, 'w') as file_obj:
        json.dump(serializable, file_obj, indent=2)


def _read_metadata(metadata_path):
    with open(metadata_path) as file_obj:
        return json.load(file_obj)


def _require_supported_external_pix_fmt(codec_name, req_pix_fmt):
    if req_pix_fmt not in {
        'yuv420p', 'yuv420p10le', 'yuv420p12le', 'yuv420p16le',
        'yuv444p', 'yuv444p10le',
    }:
        raise AssertionError(
            f'{codec_name} wrapper currently supports only yuv420p/yuv420p10le/yuv420p12le/yuv420p16le/yuv444p/yuv444p10le intermediate formats, '
            f'got {req_pix_fmt}'
        )


def _needs_even_geometry(req_pix_fmt):
    return req_pix_fmt.startswith('yuv420')


def _pad_array_for_even_geometry(input_array, width, height):
    padded_array = input_array
    padded_width = width
    padded_height = height

    if width % 2 != 0:
        padded_array = np.concatenate([padded_array, padded_array[:, -1:, :, :]], axis=1)
        padded_width += 1

    if height % 2 != 0:
        padded_array = np.concatenate([padded_array, padded_array[:, :, -1:, :]], axis=2)
        padded_height += 1

    return padded_array, padded_width, padded_height


def _crop_array_to_original_geometry(video_data, width, height):
    return video_data[:, :width, :height, :]


def _wait_for_output_file(output_path, timeout_seconds=5.0):
    output_path = Path(output_path)
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if output_path.exists() and output_path.stat().st_size > 0:
            return
        time.sleep(0.1)
    raise FileNotFoundError(f'Expected output file was not created: {output_path}')


def _iter_ffmpeg_input_frames(input_array, planar_in):
    if planar_in:
        return np.transpose(input_array, (0, 3, 2, 1))
    return np.transpose(input_array, (0, 2, 1, 3))


def _encode_with_ffmpeg_to_y4m(input_array, x, y, input_pix_fmt, req_pix_fmt, y4m_path, planar_in):
    input_pipe = ffmpeg.input('pipe:', format='rawvideo', pix_fmt=input_pix_fmt, s=f'{x}x{y}', framerate=30)
    process = (
        ffmpeg
        .output(
            input_pipe,
            str(y4m_path),
            format='yuv4mpegpipe',
            pix_fmt=req_pix_fmt,
            r=30,
            strict='-1',
            loglevel='quiet',
        )
        .overwrite_output()
        .run_async(pipe_stdin=True, overwrite_output=True)
    )

    for frame in _iter_ffmpeg_input_frames(input_array, planar_in):
        process.stdin.write(frame.tobytes())

    process.stdin.close()
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f'ffmpeg failed while preparing {y4m_path}')


def _encode_with_ffmpeg_to_rawvideo(input_array, x, y, input_pix_fmt, req_pix_fmt, rawvideo_path, planar_in):
    input_pipe = ffmpeg.input('pipe:', format='rawvideo', pix_fmt=input_pix_fmt, s=f'{x}x{y}', framerate=30)
    process = (
        ffmpeg
        .output(
            input_pipe,
            str(rawvideo_path),
            format='rawvideo',
            pix_fmt=req_pix_fmt,
            r=30,
            loglevel='quiet',
        )
        .overwrite_output()
        .run_async(pipe_stdin=True, overwrite_output=True)
    )

    for frame in _iter_ffmpeg_input_frames(input_array, planar_in):
        process.stdin.write(frame.tobytes())

    process.stdin.close()
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f'ffmpeg failed while preparing {rawvideo_path}')


def _decode_yuv_or_y4m_to_array(decoded_path, output_pix_fmt, bits, channels, num_frames):
    probe = ffmpeg.probe(str(decoded_path))
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])

    process = (
        ffmpeg
        .input(str(decoded_path))
        .output('pipe:', format='rawvideo', pix_fmt=output_pix_fmt, loglevel='quiet')
        .run_async(pipe_stdout=True)
    )
    dtype = np.uint8 if bits == 8 else np.uint16
    data = np.frombuffer(process.stdout.read(), dtype)
    process.stdout.close()
    process.wait()
    output_shape = [num_frames, channels, height, width]
    assert len(data) == np.prod(output_shape), f'{decoded_path} produced unexpected decoded size'
    video_data = data.reshape(output_shape)
    return np.transpose(video_data, (0, 3, 2, 1))


def _decode_rawvideo_to_array(decoded_path, input_pix_fmt, output_pix_fmt, bits, channels, num_frames, width, height):
    process = (
        ffmpeg
        .input(
            str(decoded_path),
            format='rawvideo',
            pix_fmt=input_pix_fmt,
            s=f'{width}x{height}',
            framerate=30,
        )
        .output('pipe:', format='rawvideo', pix_fmt=output_pix_fmt, loglevel='quiet')
        .run_async(pipe_stdout=True)
    )
    dtype = np.uint8 if bits == 8 else np.uint16
    data = np.frombuffer(process.stdout.read(), dtype)
    process.stdout.close()
    process.wait()
    output_shape = [num_frames, channels, height, width]
    assert len(data) == np.prod(output_shape), f'{decoded_path} produced unexpected decoded size'
    video_data = data.reshape(output_shape)
    return np.transpose(video_data, (0, 3, 2, 1))


def _build_vvenc_command(encoder_bin, input_path, output_path, params, metadata):
    cmd = [encoder_bin, '-i', str(input_path), '-o', str(output_path)]
    preset = params.get('preset', 'medium')
    qp = params.get('qp', params.get('crf', 32))
    threads = params.get('threads', params.get('t'))
    cmd += ['--preset', str(preset), '-q', str(qp)]
    if threads is not None:
        cmd += ['-t', str(threads)]
    if params.get('qpa', None) is not None:
        cmd += ['--qpa', str(params['qpa'])]
    if metadata['BITS'] == 10:
        cmd += ['--additional', 'InternalBitDepth=10']
    user_args = params.get('cli_args', [])
    cmd += [str(arg) for arg in user_args]
    return cmd


def _build_uavs3e_command(encoder_bin, input_path, output_path, params, metadata):
    cmd = [
        encoder_bin,
        '-i', str(input_path),
        '-o', str(output_path),
        '-w', str(metadata['WIDTH']),
        '-h', str(metadata['HEIGHT']),
        '-f', str(metadata['FRAMES']),
        '--fps_num', str(params.get('fps_num', 30)),
        '--fps_den', str(params.get('fps_den', 1)),
        '-d', str(metadata['BITS']),
        '--internal_bit_depth', str(metadata['BITS']),
        '-q', str(params.get('qp', params.get('crf', 32))),
    ]

    i_period = params.get('i_period')
    if i_period is not None:
        cmd += ['-p', str(i_period)]

    max_b_frames = params.get('max_b_frames')
    if max_b_frames is not None:
        cmd += ['-g', str(max_b_frames)]

    threads = params.get('threads', params.get('t'))
    if threads is not None:
        cmd += ['--wpp_threads', str(threads)]

    frame_threads = params.get('frm_threads')
    if frame_threads is not None:
        cmd += ['--frm_threads', str(frame_threads)]

    verbose = params.get('verbose')
    if verbose is not None:
        cmd += ['-v', str(verbose)]

    user_args = params.get('cli_args', [])
    cmd += [str(arg) for arg in user_args]
    return cmd


def _find_vtm_repo_root(encoder_bin):
    encoder_path = Path(encoder_bin).resolve()
    for parent in encoder_path.parents:
        if (parent / 'cfg' / 'encoder_intra_vtm.cfg').exists():
            return parent
    raise FileNotFoundError(f'Could not locate VTM cfg directory relative to {encoder_bin}')


def _build_vtm_command(encoder_bin, input_path, output_path, params, metadata):
    repo_root = _find_vtm_repo_root(encoder_bin)
    temp_cfg = Path(input_path).with_suffix('.cfg')
    qp = params.get('qp', params.get('crf', 32))
    fps_num = int(params.get('fps_num', 30))
    fps_den = int(params.get('fps_den', 1))
    frame_rate = max(1, fps_num // max(1, fps_den))
    intra_period = params.get('intra_period', 1)
    gop_size = params.get('gop_size', 1)
    wavefront_synchro = int(params.get('wavefront_synchro', 1))
    temp_cfg.write_text(
        '\n'.join([
            f'InputFile                     : {input_path}',
            f'BitstreamFile                 : {output_path}',
            f'ReconFile                     : {Path(input_path).with_name("recon.yuv")}',
            f'InputBitDepth                 : {metadata["BITS"]}',
            f'InternalBitDepth              : {metadata["BITS"]}',
            f'OutputBitDepth                : {metadata["BITS"]}',
            'InputChromaFormat             : 420',
            f'FrameRate                     : {frame_rate}',
            'FrameSkip                     : 0',
            'TemporalSubsampleRatio        : 1',
            f'SourceWidth                   : {metadata["WIDTH"]}',
            f'SourceHeight                  : {metadata["HEIGHT"]}',
            f'FramesToBeEncoded             : {metadata["FRAMES"]}',
            f'IntraPeriod                   : {intra_period}',
            'DecodingRefreshType           : 1',
            f'GOPSize                       : {gop_size}',
            f'WaveFrontSynchro              : {wavefront_synchro}',
            f'QP                            : {qp}',
            'Profile                       : none',
            'Level                         : none',
        ]) + '\n',
        encoding='ascii',
    )
    cmd = [
        encoder_bin,
        '-c', str(repo_root / 'cfg' / 'encoder_intra_vtm.cfg'),
        '-c', str(temp_cfg),
    ]
    return cmd


def _build_external_encode_command(codec_name, encoder_bin, input_path, output_path, params, metadata):
    if codec_name == 'vvenc':
        return _build_vvenc_command(encoder_bin, input_path, output_path, params, metadata)
    if codec_name == 'uavs3e':
        return _build_uavs3e_command(encoder_bin, input_path, output_path, params, metadata)
    if codec_name == 'vtm':
        return _build_vtm_command(encoder_bin, input_path, output_path, params, metadata)
    raise AssertionError(f'Unsupported external codec {codec_name}')


def _build_external_decode_command(codec_name, decoder_bin, input_path, output_path, params):
    if codec_name == 'vvenc':
        cmd = [decoder_bin, '-b', str(input_path), '-o', str(output_path)]
        threads = params.get('threads', params.get('t'))
        if threads is not None:
            cmd += ['-t', str(threads)]
        return cmd
    if codec_name == 'uavs3e':
        cmd = [decoder_bin, '-i', str(input_path), '-o', str(output_path)]
        threads = params.get('threads', params.get('t'))
        if threads is not None:
            cmd += ['-t', str(threads)]
        return cmd
    if codec_name == 'vtm':
        return [decoder_bin, '-b', str(input_path), '-o', str(output_path)]
    raise AssertionError(f'Unsupported external codec {codec_name}')


def _external_video_write(video_path, array, x, y, output_params, planar_in=True,
                          input_pix_fmt='gbrp', loglevel='quiet', metadata=None):
    metadata = {} if metadata is None else dict(metadata)
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    codec_name = output_params['c:v']
    req_pix_fmt = output_params['pix_fmt']
    _require_supported_external_pix_fmt(codec_name, req_pix_fmt)

    encoder_bin = output_params.get('encoder_bin') or _find_binary(EXTERNAL_VIDEO_CODECS[codec_name]['encoder_bins'])
    if encoder_bin is None:
        raise FileNotFoundError(
            f'Could not find an encoder for {codec_name}. Checked {EXTERNAL_VIDEO_CODECS[codec_name]["encoder_bins"]}'
        )

    metadata['BACKEND'] = codec_name
    metadata['ORIG_WIDTH'] = x
    metadata['ORIG_HEIGHT'] = y

    array_to_encode = array
    encode_width = x
    encode_height = y
    if _needs_even_geometry(req_pix_fmt) and (x % 2 != 0 or y % 2 != 0):
        array_to_encode, encode_width, encode_height = _pad_array_for_even_geometry(array, x, y)

    metadata['WIDTH'] = encode_width
    metadata['HEIGHT'] = encode_height
    metadata_path = _metadata_path(video_path)
    _write_metadata(metadata_path, metadata)

    with tempfile.TemporaryDirectory(prefix=f'{codec_name}_encode_') as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        if codec_name in {'uavs3e', 'vtm'}:
            staged_input_path = temp_dir / 'input.yuv'
            _encode_with_ffmpeg_to_rawvideo(array_to_encode, encode_width, encode_height, input_pix_fmt, req_pix_fmt, staged_input_path, planar_in)
        else:
            staged_input_path = temp_dir / 'input.y4m'
            _encode_with_ffmpeg_to_y4m(array_to_encode, encode_width, encode_height, input_pix_fmt, req_pix_fmt, staged_input_path, planar_in)
        cmd = _build_external_encode_command(codec_name, encoder_bin, staged_input_path, video_path, output_params, metadata)
        completed = subprocess.run(cmd, capture_output=loglevel == 'quiet', text=True)
        if completed.returncode != 0:
            stderr = completed.stderr.strip() if completed.stderr else ''
            raise RuntimeError(f'{codec_name} encoding failed for {video_path}: {stderr}')
        _wait_for_output_file(video_path)


def _external_video_read(video_path, loglevel='quiet'):
    metadata_path = _metadata_path(video_path)
    meta_info = _read_metadata(metadata_path)
    codec_name = meta_info['BACKEND']
    decoder_candidates = EXTERNAL_VIDEO_CODECS[codec_name]['decoder_bins']
    decoder_bin = meta_info.get('DECODER_BIN') or _find_binary(decoder_candidates)
    if decoder_bin is None:
        raise FileNotFoundError(f'Could not find a decoder for {codec_name}. Checked {decoder_candidates}')

    with tempfile.TemporaryDirectory(prefix=f'{codec_name}_decode_') as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        decoded_path = temp_dir / ('decoded.yuv' if codec_name in {'uavs3e', 'vtm'} else 'decoded.y4m')
        cmd = _build_external_decode_command(codec_name, decoder_bin, video_path, decoded_path, meta_info)
        completed = subprocess.run(cmd, capture_output=loglevel == 'quiet', text=True)
        if completed.returncode != 0:
            stderr = completed.stderr.strip() if completed.stderr else ''
            raise RuntimeError(f'{codec_name} decoding failed for {video_path}: {stderr}')
        if codec_name in {'uavs3e', 'vtm'}:
            video_data = _decode_rawvideo_to_array(
                decoded_path,
                input_pix_fmt=meta_info['REQ_PIX_FMT'],
                output_pix_fmt=meta_info['OUT_PIX_FMT'],
                bits=int(meta_info['BITS']),
                channels=int(meta_info['CHANNELS']),
                num_frames=int(meta_info['FRAMES']),
                width=int(meta_info['WIDTH']),
                height=int(meta_info['HEIGHT']),
            )
        else:
            video_data = _decode_yuv_or_y4m_to_array(
                decoded_path,
                output_pix_fmt=meta_info['OUT_PIX_FMT'],
                bits=int(meta_info['BITS']),
                channels=int(meta_info['CHANNELS']),
                num_frames=int(meta_info['FRAMES']),
            )
    if 'ORIG_WIDTH' in meta_info and 'ORIG_HEIGHT' in meta_info:
        orig_width = int(meta_info['ORIG_WIDTH'])
        orig_height = int(meta_info['ORIG_HEIGHT'])
        if video_data.shape[1] != orig_width or video_data.shape[2] != orig_height:
            video_data = _crop_array_to_original_geometry(video_data, orig_width, orig_height)
    return video_data, meta_info