//! Decoding (symphonia, plus libopus for Opus → mono f32) and resampling (rubato).

use std::{fs::File, path::Path, sync::OnceLock};

use anyhow::Context;
use rubato::{FftFixedIn, Resampler};
use symphonia::core::{
    audio::SampleBuffer,
    codecs::{CodecRegistry, DecoderOptions},
    errors::Error as SymErr,
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};

/// symphonia's codecs, plus Opus when the `opus` feature is on.
fn codecs() -> &'static CodecRegistry {
    static REGISTRY: OnceLock<CodecRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut r = CodecRegistry::new();
        symphonia::default::register_enabled_codecs(&mut r);
        #[cfg(feature = "opus")]
        r.register_all::<symphonia_adapter_libopus::OpusDecoder>();
        r
    })
}

/// Decode any symphonia-supported file into mono f32 samples + sample rate.
pub fn decode_mono(path: &Path) -> anyhow::Result<(Vec<f32>, u32)> {
    let mss = MediaSourceStream::new(Box::new(File::open(path)?), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let mut format = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )?
        .format;
    let track = format.default_track().context("no audio track")?;
    let (track_id, params) = (track.id, track.codec_params.clone());
    let sr = params.sample_rate.context("unknown sample rate")?;
    let mut decoder = codecs().make(&params, &DecoderOptions::default())?;

    let mut out = Vec::new();
    let mut bad_frames = 0usize;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymErr::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            // a damaged frame (common in real MP3s) is skipped, like players do
            Err(SymErr::DecodeError(_)) => {
                bad_frames += 1;
                continue;
            }
            Err(e) => return Err(e.into()),
        };
        let ch = decoded.spec().channels.count();
        let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        buf.copy_interleaved_ref(decoded);
        out.extend(
            buf.samples()
                .chunks_exact(ch)
                .map(|f| f.iter().sum::<f32>() / ch as f32),
        );
    }
    anyhow::ensure!(
        !out.is_empty(),
        "no decodable audio ({bad_frames} damaged frames)"
    );
    Ok((out, sr))
}

/// FFT-based resampling of a whole mono buffer.
pub fn resample(x: &[f32], from: u32, to: u32) -> anyhow::Result<Vec<f32>> {
    if from == to {
        return Ok(x.to_vec());
    }
    let mut r = FftFixedIn::<f32>::new(from as usize, to as usize, 1024, 2, 1)?;
    let delay = r.output_delay();
    let mut out = Vec::with_capacity(x.len() * to as usize / from as usize + 2048);
    let mut pos = 0;
    while pos + r.input_frames_next() <= x.len() {
        let n = r.input_frames_next();
        out.extend_from_slice(&r.process(&[&x[pos..pos + n]], None)?[0]);
        pos += n;
    }
    out.extend_from_slice(&r.process_partial(Some(&[&x[pos..]]), None)?[0]);
    // flush the filter tail so the output is not shortened by the resampler delay
    out.extend_from_slice(&r.process_partial::<&[f32]>(None, None)?[0]);
    let want = (x.len() as u64 * to as u64 / from as u64) as usize;
    Ok(out.into_iter().skip(delay).take(want).collect())
}

#[cfg(test)]
mod tests {
    #[test]
    fn resample_keeps_duration_and_frequency() {
        let x: Vec<f32> = (0..22050)
            .map(|i| (2.0 * std::f32::consts::PI * 441.0 * i as f32 / 22050.0).sin())
            .collect();
        let y = super::resample(&x, 22050, 48000).unwrap();
        assert_eq!(y.len(), 48000);
        // zero crossings of a 441 Hz sine over 1 s ≈ 882
        let zc = y.windows(2).filter(|w| w[0] < 0.0 && w[1] >= 0.0).count();
        assert!((430..=452).contains(&zc), "{zc}");
    }
}
