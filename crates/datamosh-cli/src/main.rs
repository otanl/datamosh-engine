// datamosh CLI test harness.
//
// Split out of the original `datamosh` crate during the workspace refactor.
// Drives the core codec (`datamosh::`) for the `raw-mosh` path and the
// `datamosh_streamfilter` crate for the elementary-stream filter path.

use std::fs::File;
use std::io::{self, Read, Write};
use std::net::UdpSocket;
use std::time::Instant;

use datamosh::{
    ActivityMode, MoshBitstreamMutationStats, MoshBitstreamParams, MoshCodec, MoshCodecConfig,
    MoshGlitchParams, MoshReferenceMode, RAW_MOSH_CONTROL_MAX, RAW_RGB_CHANNELS, RawMoshControls,
    apply_raw_mosh_controls, apply_raw_mosh_preset, raw_mosh_parameter_requires_rebuild,
    set_raw_mosh_parameter,
};
use datamosh_streamfilter::{
    Codec, Config, DatamoshStream, FrameTypeRewrite, MpegSliceDropMode, REPORT_INTERVAL,
    load_donor_stream, run_stream_inner,
};

fn main() {
    std::process::exit(run_cli(std::env::args().skip(1)));
}

pub fn run_cli(args: impl IntoIterator<Item = String>) -> i32 {
    let args: Vec<String> = args.into_iter().collect();
    if args.first().map(String::as_str) == Some("raw-mosh") {
        match parse_raw_mosh_args(args.into_iter().skip(1)) {
            Ok(Some(cli)) => {
                let stdin = io::stdin();
                let stdout = io::stdout();
                let stderr = io::stderr();

                if let Err(err) =
                    run_raw_mosh_stream(cli, stdin.lock(), stdout.lock(), stderr.lock())
                {
                    let _ = writeln!(io::stderr(), "datamosh raw-mosh: {err}");
                    1
                } else {
                    0
                }
            }
            Ok(None) => 0,
            Err(err) => {
                let _ = writeln!(io::stderr(), "datamosh raw-mosh: {err}\n");
                print_raw_mosh_help();
                2
            }
        }
    } else {
        match parse_args(args) {
            Ok(Some(cli)) => {
                let stdin = io::stdin();
                let stdout = io::stdout();
                let stderr = io::stderr();

                if let Err(err) = run_cli_stream(cli, stdin.lock(), stdout.lock(), stderr.lock()) {
                    let _ = writeln!(io::stderr(), "datamosh: {err}");
                    1
                } else {
                    0
                }
            }
            Ok(None) => 0,
            Err(err) => {
                let _ = writeln!(io::stderr(), "datamosh: {err}\n");
                print_help();
                2
            }
        }
    }
}


struct CliArgs {
    config: Config,
    donor_file: Option<String>,
}


#[derive(Debug)]
struct RawMoshCli {
    config: MoshCodecConfig,
    params: MoshGlitchParams,
    bitstream: MoshBitstreamParams,
    output_width: Option<usize>,
    output_height: Option<usize>,
    scale_mode: RawMoshScaleMode,
    control_port: Option<u16>,
    quiet: bool,
}


#[derive(Debug, Clone, Copy, Default)]
struct RawMoshControlUpdate {
    rebuild_codec: bool,
    reset_glitch_state: bool,
}


fn apply_raw_mosh_control_message(
    message: &str,
    config: &mut MoshCodecConfig,
    params: &mut MoshGlitchParams,
    bitstream: &mut MoshBitstreamParams,
    controls: &mut RawMoshControls,
) -> Result<RawMoshControlUpdate, String> {
    let mut update = RawMoshControlUpdate::default();

    for line in message.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let mut fields = line.split_whitespace();
        let Some(command) = fields.next() else {
            continue;
        };

        match command {
            "preset" => {
                let preset = fields
                    .next()
                    .ok_or_else(|| "preset control message requires a preset name".to_string())?;
                apply_raw_mosh_preset(preset, config, params, bitstream)?;
                update.rebuild_codec = true;
            }
            "set" => {
                let id = fields
                    .next()
                    .ok_or_else(|| "set control message requires a parameter id".to_string())?;
                let value = fields
                    .next()
                    .ok_or_else(|| "set control message requires a parameter value".to_string())?;
                set_raw_mosh_parameter(
                    config,
                    params,
                    bitstream,
                    id,
                    parse_f32("control value", value)?,
                )?;
                update.rebuild_codec |= raw_mosh_parameter_requires_rebuild(id);
            }
            "controls" | "control" => {
                let mut values = [
                    controls.intensity,
                    controls.motion,
                    controls.residual,
                    controls.temporal,
                    controls.bitstream,
                ];
                for slot in &mut values {
                    let Some(value) = fields.next() else {
                        break;
                    };
                    *slot = parse_f32("control amount", value)?.clamp(0.0, RAW_MOSH_CONTROL_MAX);
                }
                controls.intensity = values[0];
                controls.motion = values[1];
                controls.residual = values[2];
                controls.temporal = values[3];
                controls.bitstream = values[4];
            }
            "reset-controls" | "reset_controls" => {
                *controls = RawMoshControls::default();
            }
            "reset" | "reset-glitch" | "reset_glitch" | "clear-history" | "clear_history" => {
                update.reset_glitch_state = true;
            }
            "ping" => {}
            _ => {
                if let Some((id, value)) = line.split_once('=') {
                    let id = id.trim();
                    set_raw_mosh_parameter(
                        config,
                        params,
                        bitstream,
                        id,
                        parse_f32("control value", value.trim())?,
                    )?;
                    update.rebuild_codec |= raw_mosh_parameter_requires_rebuild(id);
                } else {
                    return Err(format!("unknown raw-mosh control command `{command}`"));
                }
            }
        }
    }

    Ok(update)
}


#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum RawMoshScaleMode {
    Nearest,
    Linear,
}


impl RawMoshScaleMode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "nearest" | "neighbor" | "point" => Ok(Self::Nearest),
            "linear" | "bilinear" => Ok(Self::Linear),
            _ => Err(format!(
                "unsupported raw-mosh scale mode `{value}`; expected nearest or linear"
            )),
        }
    }
}


fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Option<CliArgs>, String> {
    let mut args = args.into_iter().peekable();

    match args.peek().map(String::as_str) {
        Some("help" | "-h" | "--help") => {
            print_help();
            return Ok(None);
        }
        Some("filter") => {
            args.next();
        }
        Some(cmd) if cmd.starts_with('-') => {}
        Some(cmd) => return Err(format!("unknown command `{cmd}`")),
        None => {}
    }

    let mut config = Config::default();
    let mut donor_file = None;
    while let Some(arg) = args.next() {
        if let Some(value) = arg.strip_prefix("--codec=") {
            config.codec = Codec::parse(value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--donor-file=") {
            donor_file = Some(value.to_string());
            continue;
        }
        if let Some(value) = arg
            .strip_prefix("--drop-keyframe-after=")
            .or_else(|| arg.strip_prefix("--drop-idr-after="))
        {
            config.drop_idr_after = parse_u64("--drop-keyframe-after", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--recover-every=") {
            config.recover_every = parse_u64("--recover-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--drop-slice-every=") {
            config.drop_slice_every = parse_u64("--drop-slice-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--damage-slice-every=") {
            config.damage_slice_every = parse_u64("--damage-slice-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--damage-amount=") {
            config.damage_amount = parse_usize("--damage-amount", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--truncate-slice-every=") {
            config.truncate_slice_every = parse_u64("--truncate-slice-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--truncate-amount=") {
            config.truncate_amount = parse_usize("--truncate-amount", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--scramble-slice-every=") {
            config.scramble_slice_every = parse_u64("--scramble-slice-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--scramble-amount=") {
            config.scramble_amount = parse_usize("--scramble-amount", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--rotate-slice-every=") {
            config.rotate_slice_every = parse_u64("--rotate-slice-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--rotate-amount=") {
            config.rotate_amount = parse_usize("--rotate-amount", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--splice-slice-every=") {
            config.splice_slice_every = parse_u64("--splice-slice-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--splice-amount=") {
            config.splice_amount = parse_usize("--splice-amount", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--grow-slice-every=") {
            config.grow_slice_every = parse_u64("--grow-slice-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--grow-amount=") {
            config.grow_amount = parse_usize("--grow-amount", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--donor-bank-size=") {
            config.donor_bank_size = parse_usize("--donor-bank-size", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--donor-splice-slice-every=") {
            config.donor_splice_slice_every = parse_u64("--donor-splice-slice-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--donor-splice-amount=") {
            config.donor_splice_amount = parse_usize("--donor-splice-amount", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--donor-grow-slice-every=") {
            config.donor_grow_slice_every = parse_u64("--donor-grow-slice-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--donor-grow-amount=") {
            config.donor_grow_amount = parse_usize("--donor-grow-amount", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--donor-xor-slice-every=") {
            config.donor_xor_slice_every = parse_u64("--donor-xor-slice-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--donor-xor-amount=") {
            config.donor_xor_amount = parse_usize("--donor-xor-amount", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--donor-replace-slice-every=") {
            config.donor_replace_slice_every = parse_u64("--donor-replace-slice-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--rewrite-frame-type-every=") {
            config.rewrite_frame_type_every = parse_u64("--rewrite-frame-type-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--rewrite-frame-type-to=") {
            config.rewrite_frame_type_to = FrameTypeRewrite::parse(value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--shift-slice-address-every=") {
            config.shift_slice_address_every = parse_u64("--shift-slice-address-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--shift-slice-address-by=") {
            config.shift_slice_address_by = parse_i16("--shift-slice-address-by", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--drop-mpeg-slice-address-every=") {
            config.drop_mpeg_slice_address_every =
                parse_u8("--drop-mpeg-slice-address-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--drop-mpeg-slice-address-phase=") {
            config.drop_mpeg_slice_address_phase =
                parse_u8("--drop-mpeg-slice-address-phase", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--drop-mpeg-slice-address-mode=") {
            config.drop_mpeg_slice_address_mode = MpegSliceDropMode::parse(value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--xor-slice-every=") {
            config.xor_slice_every = parse_u64("--xor-slice-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--xor-amount=") {
            config.xor_amount = parse_usize("--xor-amount", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--echo-slice-every=") {
            config.echo_slice_every = parse_u64("--echo-slice-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--echo-count=") {
            config.echo_count = parse_u64("--echo-count", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--replace-slice-every=") {
            config.replace_slice_every = parse_u64("--replace-slice-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--repeat-slice-every=") {
            config.repeat_slice_every = parse_u64("--repeat-slice-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--repeat-count=") {
            config.repeat_count = parse_u64("--repeat-count", value)?;
            continue;
        }

        match arg.as_str() {
            "--codec" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--codec requires a value".to_string())?;
                config.codec = Codec::parse(&value)?;
            }
            "--donor-file" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--donor-file requires a value".to_string())?;
                donor_file = Some(value);
            }
            "--drop-keyframe-after" | "--drop-idr-after" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--drop-keyframe-after requires a value".to_string())?;
                config.drop_idr_after = parse_u64("--drop-keyframe-after", &value)?;
            }
            "--recover-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--recover-every requires a value".to_string())?;
                config.recover_every = parse_u64("--recover-every", &value)?;
            }
            "--drop-slice-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--drop-slice-every requires a value".to_string())?;
                config.drop_slice_every = parse_u64("--drop-slice-every", &value)?;
            }
            "--damage-slice-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--damage-slice-every requires a value".to_string())?;
                config.damage_slice_every = parse_u64("--damage-slice-every", &value)?;
            }
            "--damage-amount" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--damage-amount requires a value".to_string())?;
                config.damage_amount = parse_usize("--damage-amount", &value)?;
            }
            "--truncate-slice-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--truncate-slice-every requires a value".to_string())?;
                config.truncate_slice_every = parse_u64("--truncate-slice-every", &value)?;
            }
            "--truncate-amount" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--truncate-amount requires a value".to_string())?;
                config.truncate_amount = parse_usize("--truncate-amount", &value)?;
            }
            "--scramble-slice-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--scramble-slice-every requires a value".to_string())?;
                config.scramble_slice_every = parse_u64("--scramble-slice-every", &value)?;
            }
            "--scramble-amount" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--scramble-amount requires a value".to_string())?;
                config.scramble_amount = parse_usize("--scramble-amount", &value)?;
            }
            "--rotate-slice-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--rotate-slice-every requires a value".to_string())?;
                config.rotate_slice_every = parse_u64("--rotate-slice-every", &value)?;
            }
            "--rotate-amount" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--rotate-amount requires a value".to_string())?;
                config.rotate_amount = parse_usize("--rotate-amount", &value)?;
            }
            "--splice-slice-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--splice-slice-every requires a value".to_string())?;
                config.splice_slice_every = parse_u64("--splice-slice-every", &value)?;
            }
            "--splice-amount" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--splice-amount requires a value".to_string())?;
                config.splice_amount = parse_usize("--splice-amount", &value)?;
            }
            "--grow-slice-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--grow-slice-every requires a value".to_string())?;
                config.grow_slice_every = parse_u64("--grow-slice-every", &value)?;
            }
            "--grow-amount" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--grow-amount requires a value".to_string())?;
                config.grow_amount = parse_usize("--grow-amount", &value)?;
            }
            "--donor-bank-size" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--donor-bank-size requires a value".to_string())?;
                config.donor_bank_size = parse_usize("--donor-bank-size", &value)?;
            }
            "--donor-splice-slice-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--donor-splice-slice-every requires a value".to_string())?;
                config.donor_splice_slice_every = parse_u64("--donor-splice-slice-every", &value)?;
            }
            "--donor-splice-amount" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--donor-splice-amount requires a value".to_string())?;
                config.donor_splice_amount = parse_usize("--donor-splice-amount", &value)?;
            }
            "--donor-grow-slice-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--donor-grow-slice-every requires a value".to_string())?;
                config.donor_grow_slice_every = parse_u64("--donor-grow-slice-every", &value)?;
            }
            "--donor-grow-amount" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--donor-grow-amount requires a value".to_string())?;
                config.donor_grow_amount = parse_usize("--donor-grow-amount", &value)?;
            }
            "--donor-xor-slice-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--donor-xor-slice-every requires a value".to_string())?;
                config.donor_xor_slice_every = parse_u64("--donor-xor-slice-every", &value)?;
            }
            "--donor-xor-amount" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--donor-xor-amount requires a value".to_string())?;
                config.donor_xor_amount = parse_usize("--donor-xor-amount", &value)?;
            }
            "--donor-replace-slice-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--donor-replace-slice-every requires a value".to_string())?;
                config.donor_replace_slice_every =
                    parse_u64("--donor-replace-slice-every", &value)?;
            }
            "--rewrite-frame-type-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--rewrite-frame-type-every requires a value".to_string())?;
                config.rewrite_frame_type_every = parse_u64("--rewrite-frame-type-every", &value)?;
            }
            "--rewrite-frame-type-to" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--rewrite-frame-type-to requires a value".to_string())?;
                config.rewrite_frame_type_to = FrameTypeRewrite::parse(&value)?;
            }
            "--shift-slice-address-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--shift-slice-address-every requires a value".to_string())?;
                config.shift_slice_address_every =
                    parse_u64("--shift-slice-address-every", &value)?;
            }
            "--shift-slice-address-by" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--shift-slice-address-by requires a value".to_string())?;
                config.shift_slice_address_by = parse_i16("--shift-slice-address-by", &value)?;
            }
            "--drop-mpeg-slice-address-every" => {
                let value = args.next().ok_or_else(|| {
                    "--drop-mpeg-slice-address-every requires a value".to_string()
                })?;
                config.drop_mpeg_slice_address_every =
                    parse_u8("--drop-mpeg-slice-address-every", &value)?;
            }
            "--drop-mpeg-slice-address-phase" => {
                let value = args.next().ok_or_else(|| {
                    "--drop-mpeg-slice-address-phase requires a value".to_string()
                })?;
                config.drop_mpeg_slice_address_phase =
                    parse_u8("--drop-mpeg-slice-address-phase", &value)?;
            }
            "--drop-mpeg-slice-address-mode" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--drop-mpeg-slice-address-mode requires a value".to_string())?;
                config.drop_mpeg_slice_address_mode = MpegSliceDropMode::parse(&value)?;
            }
            "--xor-slice-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--xor-slice-every requires a value".to_string())?;
                config.xor_slice_every = parse_u64("--xor-slice-every", &value)?;
            }
            "--xor-amount" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--xor-amount requires a value".to_string())?;
                config.xor_amount = parse_usize("--xor-amount", &value)?;
            }
            "--echo-slice-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--echo-slice-every requires a value".to_string())?;
                config.echo_slice_every = parse_u64("--echo-slice-every", &value)?;
            }
            "--echo-count" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--echo-count requires a value".to_string())?;
                config.echo_count = parse_u64("--echo-count", &value)?;
            }
            "--replace-slice-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--replace-slice-every requires a value".to_string())?;
                config.replace_slice_every = parse_u64("--replace-slice-every", &value)?;
            }
            "--repeat-slice-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--repeat-slice-every requires a value".to_string())?;
                config.repeat_slice_every = parse_u64("--repeat-slice-every", &value)?;
            }
            "--repeat-count" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--repeat-count requires a value".to_string())?;
                config.repeat_count = parse_u64("--repeat-count", &value)?;
            }
            "--drop-headers-after-first" => {
                config.drop_headers_after_first = true;
            }
            "--quiet" => {
                config.quiet = true;
            }
            "-h" | "--help" => {
                print_help();
                return Ok(None);
            }
            _ => return Err(format!("unknown option `{arg}`")),
        }
    }

    Ok(Some(CliArgs { config, donor_file }))
}


fn parse_raw_mosh_args(
    args: impl IntoIterator<Item = String>,
) -> Result<Option<RawMoshCli>, String> {
    let mut args = args.into_iter().peekable();
    if matches!(
        args.peek().map(String::as_str),
        Some("help" | "-h" | "--help")
    ) {
        print_raw_mosh_help();
        return Ok(None);
    }

    let mut width = None;
    let mut height = None;
    let mut output_width = None;
    let mut output_height = None;
    let mut upscale = None;
    let mut scale_mode = RawMoshScaleMode::Nearest;
    let mut config = MoshCodecConfig::default();
    let mut params = MoshGlitchParams::default();
    let mut bitstream = MoshBitstreamParams::default();
    let mut control_port = None;
    let mut quiet = false;

    while let Some(arg) = args.next() {
        if let Some(value) = arg.strip_prefix("--preset=") {
            apply_raw_mosh_preset(value, &mut config, &mut params, &mut bitstream)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--width=") {
            width = Some(parse_usize("--width", value)?);
            continue;
        }
        if let Some(value) = arg.strip_prefix("--height=") {
            height = Some(parse_usize("--height", value)?);
            continue;
        }
        if let Some(value) = arg.strip_prefix("--output-width=") {
            output_width = Some(parse_usize("--output-width", value)?);
            continue;
        }
        if let Some(value) = arg.strip_prefix("--output-height=") {
            output_height = Some(parse_usize("--output-height", value)?);
            continue;
        }
        if let Some(value) = arg.strip_prefix("--upscale=") {
            upscale = Some(parse_usize("--upscale", value)?);
            continue;
        }
        if let Some(value) = arg.strip_prefix("--scale-mode=") {
            scale_mode = RawMoshScaleMode::parse(value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--control-port=") {
            control_port = Some(parse_u16("--control-port", value)?);
            continue;
        }
        if let Some(value) = arg.strip_prefix("--block-size=") {
            config.block_size = parse_usize("--block-size", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--search-radius=") {
            config.search_radius = parse_i16("--search-radius", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--search-step=") {
            config.search_step = parse_i16("--search-step", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--keyframe-every=") {
            config.keyframe_interval = parse_u64("--keyframe-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--history=") {
            config.history_len = parse_usize("--history", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--reference-mode=") {
            config.reference_mode = MoshReferenceMode::parse(value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--mv-scale=") {
            let scale = parse_f32("--mv-scale", value)?;
            params.mv_scale_x = scale;
            params.mv_scale_y = scale;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--mv-scale-x=") {
            params.mv_scale_x = parse_f32("--mv-scale-x", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--mv-scale-y=") {
            params.mv_scale_y = parse_f32("--mv-scale-y", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--mv-jitter=") {
            params.mv_jitter = parse_i16("--mv-jitter", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--mv-quant=") {
            params.mv_quant = parse_i16("--mv-quant", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--reference-lag=") {
            params.reference_lag = parse_usize("--reference-lag", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--residual-keep=") {
            params.residual_keep = parse_f32("--residual-keep", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--residual-invert-every=") {
            params.residual_invert_every = parse_u64("--residual-invert-every", value)?;
            continue;
        }
        if let Some(value) = arg
            .strip_prefix("--residual-address-shift-x=")
            .or_else(|| arg.strip_prefix("--residual-drift-x="))
        {
            params.residual_address_shift_x = parse_i16("--residual-address-shift-x", value)?;
            continue;
        }
        if let Some(value) = arg
            .strip_prefix("--residual-address-shift-y=")
            .or_else(|| arg.strip_prefix("--residual-drift-y="))
        {
            params.residual_address_shift_y = parse_i16("--residual-address-shift-y", value)?;
            continue;
        }
        if let Some(value) = arg
            .strip_prefix("--residual-address-jitter=")
            .or_else(|| arg.strip_prefix("--residual-jitter="))
        {
            params.residual_address_jitter = parse_i16("--residual-address-jitter", value)?;
            continue;
        }
        if let Some(value) = arg
            .strip_prefix("--residual-channel-shift=")
            .or_else(|| arg.strip_prefix("--residual-channel-rotate="))
        {
            params.residual_channel_shift = parse_i16("--residual-channel-shift", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--temporal-slice-height=") {
            params.temporal_slice_height = parse_usize("--temporal-slice-height", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--temporal-slice-lag-span=") {
            params.temporal_slice_lag_span = parse_usize("--temporal-slice-lag-span", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--temporal-slice-drift=") {
            params.temporal_slice_drift = parse_i16("--temporal-slice-drift", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--residual-bank-size=") {
            params.residual_bank_size = parse_usize("--residual-bank-size", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--residual-bank-stride=") {
            params.residual_bank_stride = parse_i32("--residual-bank-stride", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--residual-bank-shuffle-every=") {
            params.residual_bank_shuffle_every = parse_u64("--residual-bank-shuffle-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--reference-channel-shift=") {
            params.reference_channel_shift = parse_i16("--reference-channel-shift", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--reference-channel-lag-span=") {
            params.reference_channel_lag_span = parse_usize("--reference-channel-lag-span", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--reference-channel-lag-stride=") {
            params.reference_channel_lag_stride =
                parse_i16("--reference-channel-lag-stride", value)?;
            continue;
        }
        if let Some(value) = arg
            .strip_prefix("--mv-bank-size=")
            .or_else(|| arg.strip_prefix("--vector-bank-size="))
        {
            params.mv_bank_size = parse_usize("--mv-bank-size", value)?;
            continue;
        }
        if let Some(value) = arg
            .strip_prefix("--mv-bank-stride=")
            .or_else(|| arg.strip_prefix("--vector-bank-stride="))
        {
            params.mv_bank_stride = parse_i32("--mv-bank-stride", value)?;
            continue;
        }
        if let Some(value) = arg
            .strip_prefix("--mv-bank-shuffle-every=")
            .or_else(|| arg.strip_prefix("--vector-bank-shuffle-every="))
        {
            params.mv_bank_shuffle_every = parse_u64("--mv-bank-shuffle-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--block-remap-every=") {
            params.block_remap_every = parse_u64("--block-remap-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--block-remap-stride=") {
            params.block_remap_stride = parse_i32("--block-remap-stride", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--channel-shift=") {
            params.channel_shift = parse_i16("--channel-shift", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--activity-mode=") {
            params.activity_mode = ActivityMode::parse(value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--activity-threshold=") {
            params.activity_threshold = parse_u16("--activity-threshold", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--activity-softness=") {
            params.activity_softness = parse_u16("--activity-softness", value)?;
            continue;
        }
        if let Some(value) = arg
            .strip_prefix("--reference-bleed=")
            .or_else(|| arg.strip_prefix("--dirty-floor="))
        {
            params.reference_bleed = parse_f32("--reference-bleed", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--reference-latch-frames=") {
            params.reference_latch_frames = parse_u64("--reference-latch-frames", value)?;
            continue;
        }
        if let Some(value) = arg
            .strip_prefix("--reference-slots=")
            .or_else(|| arg.strip_prefix("--reference-slot-count="))
        {
            params.reference_slot_count = parse_usize("--reference-slots", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--reference-slot-shuffle-every=") {
            params.reference_slot_shuffle_every =
                parse_u64("--reference-slot-shuffle-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--reference-scanline-height=") {
            params.reference_scanline_height = parse_usize("--reference-scanline-height", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--reference-scanline-lag-span=") {
            params.reference_scanline_lag_span =
                parse_usize("--reference-scanline-lag-span", value)?;
            continue;
        }
        if let Some(value) = arg
            .strip_prefix("--overlap=")
            .or_else(|| arg.strip_prefix("--motion-overlap="))
        {
            params.overlap = parse_usize("--overlap", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--motion-diffusion=") {
            params.motion_diffusion = parse_f32("--motion-diffusion", value)?;
            continue;
        }
        if let Some(value) = arg
            .strip_prefix("--mv-field-interpolation=")
            .or_else(|| arg.strip_prefix("--mv-field-interp="))
        {
            params.mv_field_interpolation = parse_f32("--mv-field-interpolation", value)?;
            continue;
        }
        if let Some(value) = arg
            .strip_prefix("--sample-address-desync=")
            .or_else(|| arg.strip_prefix("--address-desync="))
        {
            params.sample_address_desync = parse_f32("--sample-address-desync", value)?;
            continue;
        }
        if let Some(value) = arg
            .strip_prefix("--pixel-grain=")
            .or_else(|| arg.strip_prefix("--glitch-cell-size="))
            .or_else(|| arg.strip_prefix("--sample-grain="))
        {
            params.glitch_cell_size = parse_usize("--pixel-grain", value)?;
            params.glitch_cell_width = 0;
            params.glitch_cell_height = 0;
            continue;
        }
        if let Some(value) = arg
            .strip_prefix("--pixel-grain-x=")
            .or_else(|| arg.strip_prefix("--glitch-cell-width="))
            .or_else(|| arg.strip_prefix("--sample-grain-x="))
        {
            params.glitch_cell_width = parse_usize("--pixel-grain-x", value)?;
            continue;
        }
        if let Some(value) = arg
            .strip_prefix("--pixel-grain-y=")
            .or_else(|| arg.strip_prefix("--glitch-cell-height="))
            .or_else(|| arg.strip_prefix("--sample-grain-y="))
        {
            params.glitch_cell_height = parse_usize("--pixel-grain-y", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--mv-predictor-desync-every=") {
            params.mv_predictor_desync_every = parse_u64("--mv-predictor-desync-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--mv-predictor-desync-x=") {
            params.mv_predictor_desync_x = parse_i16("--mv-predictor-desync-x", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--mv-predictor-desync-y=") {
            params.mv_predictor_desync_y = parse_i16("--mv-predictor-desync-y", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--bs-mv-sign-flip-every=") {
            bitstream.enabled = true;
            bitstream.mv_sign_flip_every = parse_u64("--bs-mv-sign-flip-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--bs-mv-delta-every=") {
            bitstream.enabled = true;
            bitstream.mv_delta_every = parse_u64("--bs-mv-delta-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--bs-mv-delta-x=") {
            bitstream.enabled = true;
            bitstream.mv_delta_x = parse_i16("--bs-mv-delta-x", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--bs-mv-delta-y=") {
            bitstream.enabled = true;
            bitstream.mv_delta_y = parse_i16("--bs-mv-delta-y", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--bs-block-shift-every=") {
            bitstream.enabled = true;
            bitstream.block_address_shift_every = parse_u64("--bs-block-shift-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--bs-block-shift-x=") {
            bitstream.enabled = true;
            bitstream.block_address_shift_x = parse_i16("--bs-block-shift-x", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--bs-block-shift-y=") {
            bitstream.enabled = true;
            bitstream.block_address_shift_y = parse_i16("--bs-block-shift-y", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--bs-residual-zero-every=") {
            bitstream.enabled = true;
            bitstream.residual_zero_every = parse_u64("--bs-residual-zero-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--bs-residual-xor-every=") {
            bitstream.enabled = true;
            bitstream.residual_xor_every = parse_u64("--bs-residual-xor-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--bs-residual-xor-mask=") {
            bitstream.enabled = true;
            bitstream.residual_xor_mask = parse_u8("--bs-residual-xor-mask", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--bs-entropy-slip-every=") {
            bitstream.enabled = true;
            bitstream.entropy_slip_every = parse_u64("--bs-entropy-slip-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--bs-entropy-slip-by=") {
            bitstream.enabled = true;
            bitstream.entropy_slip_bytes = parse_i16("--bs-entropy-slip-by", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--bs-entropy-resync-bytes=") {
            bitstream.enabled = true;
            bitstream.entropy_resync_bytes = parse_usize("--bs-entropy-resync-bytes", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--bs-entropy-windows=") {
            bitstream.enabled = true;
            bitstream.entropy_slip_windows = parse_usize("--bs-entropy-windows", value)?;
            continue;
        }
        if let Some(value) = arg
            .strip_prefix("--bs-coeff-every=")
            .or_else(|| arg.strip_prefix("--bs-transform-every="))
        {
            bitstream.enabled = true;
            bitstream.coeff_glitch_every = parse_u64("--bs-coeff-every", value)?;
            continue;
        }
        if let Some(value) = arg
            .strip_prefix("--bs-coeff-block-size=")
            .or_else(|| arg.strip_prefix("--bs-transform-block-size="))
        {
            bitstream.enabled = true;
            bitstream.coeff_block_size = parse_usize("--bs-coeff-block-size", value)?;
            continue;
        }
        if let Some(value) = arg
            .strip_prefix("--bs-coeff-shift=")
            .or_else(|| arg.strip_prefix("--bs-transform-shift="))
        {
            bitstream.enabled = true;
            bitstream.coeff_shift = parse_i16("--bs-coeff-shift", value)?;
            continue;
        }
        if let Some(value) = arg
            .strip_prefix("--bs-coeff-sign-flip-every=")
            .or_else(|| arg.strip_prefix("--bs-transform-sign-flip-every="))
        {
            bitstream.enabled = true;
            bitstream.coeff_sign_flip_every = parse_u64("--bs-coeff-sign-flip-every", value)?;
            continue;
        }
        if let Some(value) = arg
            .strip_prefix("--bs-coeff-zero-high=")
            .or_else(|| arg.strip_prefix("--bs-transform-zero-high="))
        {
            bitstream.enabled = true;
            bitstream.coeff_zero_high = parse_usize("--bs-coeff-zero-high", value)?;
            continue;
        }
        if let Some(value) = arg
            .strip_prefix("--bs-coeff-quant=")
            .or_else(|| arg.strip_prefix("--bs-transform-quant="))
        {
            bitstream.enabled = true;
            bitstream.coeff_quant = parse_i16("--bs-coeff-quant", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--bs-codebook-every=") {
            bitstream.enabled = true;
            bitstream.codebook_replace_every = parse_u64("--bs-codebook-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--bs-codebook-tile-size=") {
            bitstream.enabled = true;
            bitstream.codebook_tile_size = parse_usize("--bs-codebook-tile-size", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--bs-codebook-slots=") {
            bitstream.enabled = true;
            bitstream.codebook_slots = parse_usize("--bs-codebook-slots", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--bs-codebook-stride=") {
            bitstream.enabled = true;
            bitstream.codebook_stride = parse_i32("--bs-codebook-stride", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--bs-codebook-update-every=") {
            bitstream.enabled = true;
            bitstream.codebook_update_every = parse_u64("--bs-codebook-update-every", value)?;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--bs-codebook-shuffle-every=") {
            bitstream.enabled = true;
            bitstream.codebook_shuffle_every = parse_u64("--bs-codebook-shuffle-every", value)?;
            continue;
        }

        match arg.as_str() {
            "--preset" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--preset requires a value".to_string())?;
                apply_raw_mosh_preset(&value, &mut config, &mut params, &mut bitstream)?;
            }
            "--width" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--width requires a value".to_string())?;
                width = Some(parse_usize("--width", &value)?);
            }
            "--height" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--height requires a value".to_string())?;
                height = Some(parse_usize("--height", &value)?);
            }
            "--output-width" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--output-width requires a value".to_string())?;
                output_width = Some(parse_usize("--output-width", &value)?);
            }
            "--output-height" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--output-height requires a value".to_string())?;
                output_height = Some(parse_usize("--output-height", &value)?);
            }
            "--upscale" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--upscale requires a value".to_string())?;
                upscale = Some(parse_usize("--upscale", &value)?);
            }
            "--scale-mode" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--scale-mode requires a value".to_string())?;
                scale_mode = RawMoshScaleMode::parse(&value)?;
            }
            "--control-port" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--control-port requires a value".to_string())?;
                control_port = Some(parse_u16("--control-port", &value)?);
            }
            "--block-size" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--block-size requires a value".to_string())?;
                config.block_size = parse_usize("--block-size", &value)?;
            }
            "--search-radius" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--search-radius requires a value".to_string())?;
                config.search_radius = parse_i16("--search-radius", &value)?;
            }
            "--search-step" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--search-step requires a value".to_string())?;
                config.search_step = parse_i16("--search-step", &value)?;
            }
            "--keyframe-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--keyframe-every requires a value".to_string())?;
                config.keyframe_interval = parse_u64("--keyframe-every", &value)?;
            }
            "--history" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--history requires a value".to_string())?;
                config.history_len = parse_usize("--history", &value)?;
            }
            "--reference-mode" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--reference-mode requires a value".to_string())?;
                config.reference_mode = MoshReferenceMode::parse(&value)?;
            }
            "--mv-scale" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--mv-scale requires a value".to_string())?;
                let scale = parse_f32("--mv-scale", &value)?;
                params.mv_scale_x = scale;
                params.mv_scale_y = scale;
            }
            "--mv-scale-x" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--mv-scale-x requires a value".to_string())?;
                params.mv_scale_x = parse_f32("--mv-scale-x", &value)?;
            }
            "--mv-scale-y" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--mv-scale-y requires a value".to_string())?;
                params.mv_scale_y = parse_f32("--mv-scale-y", &value)?;
            }
            "--mv-jitter" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--mv-jitter requires a value".to_string())?;
                params.mv_jitter = parse_i16("--mv-jitter", &value)?;
            }
            "--mv-quant" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--mv-quant requires a value".to_string())?;
                params.mv_quant = parse_i16("--mv-quant", &value)?;
            }
            "--reference-lag" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--reference-lag requires a value".to_string())?;
                params.reference_lag = parse_usize("--reference-lag", &value)?;
            }
            "--residual-keep" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--residual-keep requires a value".to_string())?;
                params.residual_keep = parse_f32("--residual-keep", &value)?;
            }
            "--residual-invert-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--residual-invert-every requires a value".to_string())?;
                params.residual_invert_every = parse_u64("--residual-invert-every", &value)?;
            }
            "--residual-address-shift-x" | "--residual-drift-x" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                params.residual_address_shift_x = parse_i16("--residual-address-shift-x", &value)?;
            }
            "--residual-address-shift-y" | "--residual-drift-y" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                params.residual_address_shift_y = parse_i16("--residual-address-shift-y", &value)?;
            }
            "--residual-address-jitter" | "--residual-jitter" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                params.residual_address_jitter = parse_i16("--residual-address-jitter", &value)?;
            }
            "--residual-channel-shift" | "--residual-channel-rotate" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                params.residual_channel_shift = parse_i16("--residual-channel-shift", &value)?;
            }
            "--temporal-slice-height" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--temporal-slice-height requires a value".to_string())?;
                params.temporal_slice_height = parse_usize("--temporal-slice-height", &value)?;
            }
            "--temporal-slice-lag-span" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--temporal-slice-lag-span requires a value".to_string())?;
                params.temporal_slice_lag_span = parse_usize("--temporal-slice-lag-span", &value)?;
            }
            "--temporal-slice-drift" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--temporal-slice-drift requires a value".to_string())?;
                params.temporal_slice_drift = parse_i16("--temporal-slice-drift", &value)?;
            }
            "--residual-bank-size" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--residual-bank-size requires a value".to_string())?;
                params.residual_bank_size = parse_usize("--residual-bank-size", &value)?;
            }
            "--residual-bank-stride" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--residual-bank-stride requires a value".to_string())?;
                params.residual_bank_stride = parse_i32("--residual-bank-stride", &value)?;
            }
            "--residual-bank-shuffle-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--residual-bank-shuffle-every requires a value".to_string())?;
                params.residual_bank_shuffle_every =
                    parse_u64("--residual-bank-shuffle-every", &value)?;
            }
            "--reference-channel-shift" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--reference-channel-shift requires a value".to_string())?;
                params.reference_channel_shift = parse_i16("--reference-channel-shift", &value)?;
            }
            "--reference-channel-lag-span" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--reference-channel-lag-span requires a value".to_string())?;
                params.reference_channel_lag_span =
                    parse_usize("--reference-channel-lag-span", &value)?;
            }
            "--reference-channel-lag-stride" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--reference-channel-lag-stride requires a value".to_string())?;
                params.reference_channel_lag_stride =
                    parse_i16("--reference-channel-lag-stride", &value)?;
            }
            "--mv-bank-size" | "--vector-bank-size" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                params.mv_bank_size = parse_usize("--mv-bank-size", &value)?;
            }
            "--mv-bank-stride" | "--vector-bank-stride" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                params.mv_bank_stride = parse_i32("--mv-bank-stride", &value)?;
            }
            "--mv-bank-shuffle-every" | "--vector-bank-shuffle-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                params.mv_bank_shuffle_every = parse_u64("--mv-bank-shuffle-every", &value)?;
            }
            "--block-remap-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--block-remap-every requires a value".to_string())?;
                params.block_remap_every = parse_u64("--block-remap-every", &value)?;
            }
            "--block-remap-stride" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--block-remap-stride requires a value".to_string())?;
                params.block_remap_stride = parse_i32("--block-remap-stride", &value)?;
            }
            "--channel-shift" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--channel-shift requires a value".to_string())?;
                params.channel_shift = parse_i16("--channel-shift", &value)?;
            }
            "--activity-mode" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--activity-mode requires a value".to_string())?;
                params.activity_mode = ActivityMode::parse(&value)?;
            }
            "--activity-threshold" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--activity-threshold requires a value".to_string())?;
                params.activity_threshold = parse_u16("--activity-threshold", &value)?;
            }
            "--activity-softness" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--activity-softness requires a value".to_string())?;
                params.activity_softness = parse_u16("--activity-softness", &value)?;
            }
            "--reference-bleed" | "--dirty-floor" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                params.reference_bleed = parse_f32("--reference-bleed", &value)?;
            }
            "--reference-latch-frames" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--reference-latch-frames requires a value".to_string())?;
                params.reference_latch_frames = parse_u64("--reference-latch-frames", &value)?;
            }
            "--reference-slots" | "--reference-slot-count" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                params.reference_slot_count = parse_usize("--reference-slots", &value)?;
            }
            "--reference-slot-shuffle-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--reference-slot-shuffle-every requires a value".to_string())?;
                params.reference_slot_shuffle_every =
                    parse_u64("--reference-slot-shuffle-every", &value)?;
            }
            "--reference-scanline-height" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--reference-scanline-height requires a value".to_string())?;
                params.reference_scanline_height =
                    parse_usize("--reference-scanline-height", &value)?;
            }
            "--reference-scanline-lag-span" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--reference-scanline-lag-span requires a value".to_string())?;
                params.reference_scanline_lag_span =
                    parse_usize("--reference-scanline-lag-span", &value)?;
            }
            "--overlap" | "--motion-overlap" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                params.overlap = parse_usize("--overlap", &value)?;
            }
            "--motion-diffusion" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--motion-diffusion requires a value".to_string())?;
                params.motion_diffusion = parse_f32("--motion-diffusion", &value)?;
            }
            "--mv-field-interpolation" | "--mv-field-interp" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                params.mv_field_interpolation = parse_f32("--mv-field-interpolation", &value)?;
            }
            "--sample-address-desync" | "--address-desync" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                params.sample_address_desync = parse_f32("--sample-address-desync", &value)?;
            }
            "--pixel-grain" | "--glitch-cell-size" | "--sample-grain" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                params.glitch_cell_size = parse_usize("--pixel-grain", &value)?;
                params.glitch_cell_width = 0;
                params.glitch_cell_height = 0;
            }
            "--pixel-grain-x" | "--glitch-cell-width" | "--sample-grain-x" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                params.glitch_cell_width = parse_usize("--pixel-grain-x", &value)?;
            }
            "--pixel-grain-y" | "--glitch-cell-height" | "--sample-grain-y" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                params.glitch_cell_height = parse_usize("--pixel-grain-y", &value)?;
            }
            "--mv-predictor-desync-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--mv-predictor-desync-every requires a value".to_string())?;
                params.mv_predictor_desync_every =
                    parse_u64("--mv-predictor-desync-every", &value)?;
            }
            "--mv-predictor-desync-x" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--mv-predictor-desync-x requires a value".to_string())?;
                params.mv_predictor_desync_x = parse_i16("--mv-predictor-desync-x", &value)?;
            }
            "--mv-predictor-desync-y" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--mv-predictor-desync-y requires a value".to_string())?;
                params.mv_predictor_desync_y = parse_i16("--mv-predictor-desync-y", &value)?;
            }
            "--bitstream" => {
                bitstream.enabled = true;
            }
            "--bs-mv-sign-flip-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--bs-mv-sign-flip-every requires a value".to_string())?;
                bitstream.enabled = true;
                bitstream.mv_sign_flip_every = parse_u64("--bs-mv-sign-flip-every", &value)?;
            }
            "--bs-mv-delta-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--bs-mv-delta-every requires a value".to_string())?;
                bitstream.enabled = true;
                bitstream.mv_delta_every = parse_u64("--bs-mv-delta-every", &value)?;
            }
            "--bs-mv-delta-x" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--bs-mv-delta-x requires a value".to_string())?;
                bitstream.enabled = true;
                bitstream.mv_delta_x = parse_i16("--bs-mv-delta-x", &value)?;
            }
            "--bs-mv-delta-y" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--bs-mv-delta-y requires a value".to_string())?;
                bitstream.enabled = true;
                bitstream.mv_delta_y = parse_i16("--bs-mv-delta-y", &value)?;
            }
            "--bs-block-shift-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--bs-block-shift-every requires a value".to_string())?;
                bitstream.enabled = true;
                bitstream.block_address_shift_every = parse_u64("--bs-block-shift-every", &value)?;
            }
            "--bs-block-shift-x" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--bs-block-shift-x requires a value".to_string())?;
                bitstream.enabled = true;
                bitstream.block_address_shift_x = parse_i16("--bs-block-shift-x", &value)?;
            }
            "--bs-block-shift-y" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--bs-block-shift-y requires a value".to_string())?;
                bitstream.enabled = true;
                bitstream.block_address_shift_y = parse_i16("--bs-block-shift-y", &value)?;
            }
            "--bs-residual-zero-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--bs-residual-zero-every requires a value".to_string())?;
                bitstream.enabled = true;
                bitstream.residual_zero_every = parse_u64("--bs-residual-zero-every", &value)?;
            }
            "--bs-residual-xor-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--bs-residual-xor-every requires a value".to_string())?;
                bitstream.enabled = true;
                bitstream.residual_xor_every = parse_u64("--bs-residual-xor-every", &value)?;
            }
            "--bs-residual-xor-mask" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--bs-residual-xor-mask requires a value".to_string())?;
                bitstream.enabled = true;
                bitstream.residual_xor_mask = parse_u8("--bs-residual-xor-mask", &value)?;
            }
            "--bs-entropy-slip-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--bs-entropy-slip-every requires a value".to_string())?;
                bitstream.enabled = true;
                bitstream.entropy_slip_every = parse_u64("--bs-entropy-slip-every", &value)?;
            }
            "--bs-entropy-slip-by" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--bs-entropy-slip-by requires a value".to_string())?;
                bitstream.enabled = true;
                bitstream.entropy_slip_bytes = parse_i16("--bs-entropy-slip-by", &value)?;
            }
            "--bs-entropy-resync-bytes" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--bs-entropy-resync-bytes requires a value".to_string())?;
                bitstream.enabled = true;
                bitstream.entropy_resync_bytes = parse_usize("--bs-entropy-resync-bytes", &value)?;
            }
            "--bs-entropy-windows" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--bs-entropy-windows requires a value".to_string())?;
                bitstream.enabled = true;
                bitstream.entropy_slip_windows = parse_usize("--bs-entropy-windows", &value)?;
            }
            "--bs-coeff-every" | "--bs-transform-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                bitstream.enabled = true;
                bitstream.coeff_glitch_every = parse_u64("--bs-coeff-every", &value)?;
            }
            "--bs-coeff-block-size" | "--bs-transform-block-size" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                bitstream.enabled = true;
                bitstream.coeff_block_size = parse_usize("--bs-coeff-block-size", &value)?;
            }
            "--bs-coeff-shift" | "--bs-transform-shift" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                bitstream.enabled = true;
                bitstream.coeff_shift = parse_i16("--bs-coeff-shift", &value)?;
            }
            "--bs-coeff-sign-flip-every" | "--bs-transform-sign-flip-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                bitstream.enabled = true;
                bitstream.coeff_sign_flip_every = parse_u64("--bs-coeff-sign-flip-every", &value)?;
            }
            "--bs-coeff-zero-high" | "--bs-transform-zero-high" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                bitstream.enabled = true;
                bitstream.coeff_zero_high = parse_usize("--bs-coeff-zero-high", &value)?;
            }
            "--bs-coeff-quant" | "--bs-transform-quant" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                bitstream.enabled = true;
                bitstream.coeff_quant = parse_i16("--bs-coeff-quant", &value)?;
            }
            "--bs-codebook-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--bs-codebook-every requires a value".to_string())?;
                bitstream.enabled = true;
                bitstream.codebook_replace_every = parse_u64("--bs-codebook-every", &value)?;
            }
            "--bs-codebook-tile-size" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--bs-codebook-tile-size requires a value".to_string())?;
                bitstream.enabled = true;
                bitstream.codebook_tile_size = parse_usize("--bs-codebook-tile-size", &value)?;
            }
            "--bs-codebook-slots" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--bs-codebook-slots requires a value".to_string())?;
                bitstream.enabled = true;
                bitstream.codebook_slots = parse_usize("--bs-codebook-slots", &value)?;
            }
            "--bs-codebook-stride" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--bs-codebook-stride requires a value".to_string())?;
                bitstream.enabled = true;
                bitstream.codebook_stride = parse_i32("--bs-codebook-stride", &value)?;
            }
            "--bs-codebook-update-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--bs-codebook-update-every requires a value".to_string())?;
                bitstream.enabled = true;
                bitstream.codebook_update_every = parse_u64("--bs-codebook-update-every", &value)?;
            }
            "--bs-codebook-shuffle-every" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--bs-codebook-shuffle-every requires a value".to_string())?;
                bitstream.enabled = true;
                bitstream.codebook_shuffle_every =
                    parse_u64("--bs-codebook-shuffle-every", &value)?;
            }
            "--wrap-motion" => {
                params.wrap_motion = true;
            }
            "--clamp-motion" => {
                params.wrap_motion = false;
            }
            "--quiet" => {
                quiet = true;
            }
            "-h" | "--help" => {
                print_raw_mosh_help();
                return Ok(None);
            }
            _ => return Err(format!("unknown raw-mosh option `{arg}`")),
        }
    }

    config.width = width.ok_or_else(|| "--width is required for raw-mosh".to_string())?;
    config.height = height.ok_or_else(|| "--height is required for raw-mosh".to_string())?;
    if output_width.is_some() ^ output_height.is_some() {
        return Err("--output-width and --output-height must be used together".to_string());
    }
    if upscale.is_some() && (output_width.is_some() || output_height.is_some()) {
        return Err("--upscale cannot be combined with --output-width/--output-height".to_string());
    }
    if let Some(factor) = upscale {
        if factor == 0 {
            return Err("--upscale must be greater than zero".to_string());
        }
        output_width =
            Some(config.width.checked_mul(factor).ok_or_else(|| {
                "--upscale output width overflows addressable memory".to_string()
            })?);
        output_height =
            Some(config.height.checked_mul(factor).ok_or_else(|| {
                "--upscale output height overflows addressable memory".to_string()
            })?);
    }
    if output_width == Some(0) || output_height == Some(0) {
        return Err("--output-width and --output-height must be greater than zero".to_string());
    }
    Ok(Some(RawMoshCli {
        config,
        params,
        bitstream,
        output_width,
        output_height,
        scale_mode,
        control_port,
        quiet,
    }))
}


fn parse_u64(name: &str, value: &str) -> Result<u64, String> {
    value
        .parse()
        .map_err(|_| format!("{name} must be a non-negative integer"))
}


fn parse_usize(name: &str, value: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("{name} must be a non-negative integer"))
}


fn parse_i16(name: &str, value: &str) -> Result<i16, String> {
    value
        .parse()
        .map_err(|_| format!("{name} must be a signed integer"))
}


fn parse_u16(name: &str, value: &str) -> Result<u16, String> {
    value
        .parse()
        .map_err(|_| format!("{name} must be an integer from 0 to 65535"))
}


fn parse_i32(name: &str, value: &str) -> Result<i32, String> {
    value
        .parse()
        .map_err(|_| format!("{name} must be a signed integer"))
}


fn parse_f32(name: &str, value: &str) -> Result<f32, String> {
    let parsed: f32 = value
        .parse()
        .map_err(|_| format!("{name} must be a number"))?;
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(format!("{name} must be finite"))
    }
}


fn parse_u8(name: &str, value: &str) -> Result<u8, String> {
    value
        .parse()
        .map_err(|_| format!("{name} must be an integer from 0 to 255"))
}


fn raw_rgb_frame_len(width: usize, height: usize) -> io::Result<usize> {
    if width == 0 || height == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "raw RGB24 frame dimensions must be greater than zero",
        ));
    }
    width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(RAW_RGB_CHANNELS))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "raw RGB24 frame dimensions overflow addressable memory",
            )
        })
}


fn scale_rgb24_frame(
    input: &[u8],
    input_width: usize,
    input_height: usize,
    output_width: usize,
    output_height: usize,
    mode: RawMoshScaleMode,
    output: &mut [u8],
) -> io::Result<()> {
    let input_len = raw_rgb_frame_len(input_width, input_height)?;
    let output_len = raw_rgb_frame_len(output_width, output_height)?;
    if input.len() != input_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("input frame must be {input_len} bytes of rgb24"),
        ));
    }
    if output.len() != output_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("output frame must be {output_len} bytes of rgb24"),
        ));
    }

    match mode {
        RawMoshScaleMode::Nearest => scale_rgb24_frame_nearest(
            input,
            input_width,
            input_height,
            output_width,
            output_height,
            output,
        ),
        RawMoshScaleMode::Linear => scale_rgb24_frame_linear(
            input,
            input_width,
            input_height,
            output_width,
            output_height,
            output,
        ),
    }
}


fn scale_rgb24_frame_nearest(
    input: &[u8],
    input_width: usize,
    input_height: usize,
    output_width: usize,
    output_height: usize,
    output: &mut [u8],
) -> io::Result<()> {
    for out_y in 0..output_height {
        let src_y = out_y * input_height / output_height;
        for out_x in 0..output_width {
            let src_x = out_x * input_width / output_width;
            let src = (src_y * input_width + src_x) * RAW_RGB_CHANNELS;
            let dst = (out_y * output_width + out_x) * RAW_RGB_CHANNELS;
            output[dst..dst + RAW_RGB_CHANNELS]
                .copy_from_slice(&input[src..src + RAW_RGB_CHANNELS]);
        }
    }
    Ok(())
}


fn scale_rgb24_frame_linear(
    input: &[u8],
    input_width: usize,
    input_height: usize,
    output_width: usize,
    output_height: usize,
    output: &mut [u8],
) -> io::Result<()> {
    let x_den = output_width.saturating_sub(1).max(1);
    let y_den = output_height.saturating_sub(1).max(1);
    let x_range = input_width.saturating_sub(1);
    let y_range = input_height.saturating_sub(1);

    for out_y in 0..output_height {
        let src_y_fp = out_y * y_range * 256 / y_den;
        let y0 = (src_y_fp / 256).min(input_height - 1);
        let y1 = (y0 + 1).min(input_height - 1);
        let fy = (src_y_fp % 256) as u32;

        for out_x in 0..output_width {
            let src_x_fp = out_x * x_range * 256 / x_den;
            let x0 = (src_x_fp / 256).min(input_width - 1);
            let x1 = (x0 + 1).min(input_width - 1);
            let fx = (src_x_fp % 256) as u32;
            let dst = (out_y * output_width + out_x) * RAW_RGB_CHANNELS;

            for channel in 0..RAW_RGB_CHANNELS {
                let p00 = input[(y0 * input_width + x0) * RAW_RGB_CHANNELS + channel] as u32;
                let p10 = input[(y0 * input_width + x1) * RAW_RGB_CHANNELS + channel] as u32;
                let p01 = input[(y1 * input_width + x0) * RAW_RGB_CHANNELS + channel] as u32;
                let p11 = input[(y1 * input_width + x1) * RAW_RGB_CHANNELS + channel] as u32;
                let top = p00 * (256 - fx) + p10 * fx;
                let bottom = p01 * (256 - fx) + p11 * fx;
                output[dst + channel] = ((top * (256 - fy) + bottom * fy + 32_768) >> 16) as u8;
            }
        }
    }
    Ok(())
}


fn print_help() {
    println!(
        "\
datamosh - realtime compressed-video datamosh filter

Usage:
  datamosh filter [options]
  datamosh raw-mosh [raw options]
  datamosh [options]

Options:
  --codec <h264|hevc|mpeg4|mpeg1|mpeg2>
                              Input elementary stream codec. Default: h264
  --drop-keyframe-after <n>  Pass the first n keyframes, then drop keyframes. Default: 1
  --drop-idr-after <n>       Alias for --drop-keyframe-after.
  --recover-every <n>        After initial drops, pass every nth later keyframe. 0 disables recovery.
  --drop-slice-every <n>     Drop every nth predicted slice/VOP. 0 disables dropping.
  --damage-slice-every <n>   Corrupt every nth predicted payload. 0 disables payload damage.
  --damage-amount <n>        Bytes to flip in each damaged slice. Default: 4
  --truncate-slice-every <n> Truncate every nth predicted slice/VOP. 0 disables truncation.
  --truncate-amount <n>      Tail bytes to remove from truncated slices/VOPs. Default: 16
  --scramble-slice-every <n> Byte-scramble every nth predicted slice/VOP. 0 disables scrambling.
  --scramble-amount <n>      Payload bytes involved in each scramble. Default: 16
  --rotate-slice-every <n>   Rotate bytes inside every nth predicted payload. 0 disables rotation.
  --rotate-amount <n>        Payload bytes involved in each rotate. Default: 8
  --splice-slice-every <n>   Copy previous payload bytes into every nth payload. 0 disables splice.
  --splice-amount <n>        Payload bytes copied by splice. Default: 32
  --grow-slice-every <n>     Insert previous payload bytes into every nth unit. 0 disables grow.
  --grow-amount <n>          Payload bytes inserted by grow. Default: 8
  --donor-file <path>        Load a second elementary stream and use its predicted payloads as donors.
  --donor-bank-size <n>      Donor predicted unit history size. 0 disables donor storage. Default: 16
  --donor-splice-slice-every <n>
                              Copy donor payload bytes into every nth payload. 0 disables.
  --donor-splice-amount <n>  Donor payload bytes copied by splice. Default: 32
  --donor-grow-slice-every <n>
                              Insert donor payload bytes into every nth unit. 0 disables.
  --donor-grow-amount <n>    Donor payload bytes inserted by grow. Default: 8
  --donor-xor-slice-every <n>
                              XOR every nth predicted unit with a donor unit. 0 disables.
  --donor-xor-amount <n>     Donor payload bytes involved in each XOR. Default: 16
  --donor-replace-slice-every <n>
                              Replace every nth predicted unit with a donor unit. 0 disables.
  --rewrite-frame-type-every <n>
                              Rewrite every nth MPEG-4/MPEG-1/2 frame type header. 0 disables.
  --rewrite-frame-type-to <i|p|b|s|d>
                              Target frame type for MPEG frame type rewrite. Default: p
  --shift-slice-address-every <n>
                              Shift every nth MPEG-1/2 slice address. 0 disables address shift.
  --shift-slice-address-by <n>
                              Signed MPEG-1/2 slice address offset. Default: 1
  --drop-mpeg-slice-address-every <n>
                              Drop MPEG-1/2 slices whose address matches this period. 0 disables.
  --drop-mpeg-slice-address-phase <n>
                              Address phase for MPEG-1/2 partial slice drop. Default: 0
  --drop-mpeg-slice-address-mode <all|key|predicted>
                              Picture type scope for MPEG-1/2 partial slice drop. Default: all
  --xor-slice-every <n>      XOR every nth predicted unit with the previous unit. 0 disables XOR.
  --xor-amount <n>           Payload bytes involved in each XOR. Default: 16
  --echo-slice-every <n>     Inject previous predicted unit before every nth unit. 0 disables echo.
  --echo-count <n>           Previous-unit copies to inject on echo. Default: 1
  --replace-slice-every <n>  Replace every nth predicted unit with the previous unit. 0 disables replace.
  --repeat-slice-every <n>   Repeat every nth predicted slice/VOP. 0 disables repeats.
  --repeat-count <n>         Extra copies to write when repeating. Default: 1
  --drop-headers-after-first Drop repeated SPS/PPS headers after the first pair for harsher glitches.
  --quiet                    Do not print realtime stats to stderr.
  -h, --help                 Show this help.

The program reads an H.264 Annex B, MPEG-4 Visual, or MPEG-2 Video elementary stream from stdin and writes a modified stream to stdout.
"
    );
}


fn print_raw_mosh_help() {
    println!(
        "\
datamosh raw-mosh - raw RGB24 custom motion-codec glitch harness

Usage:
  datamosh raw-mosh --width <px> --height <px> [options]

Input and output are raw RGB24 video frames on stdin/stdout.

Codec options:
  --width <n>               Frame width in pixels. Required.
  --height <n>              Frame height in pixels. Required.
  --output-width <n>        Raw RGB24 output width after display scaling.
  --output-height <n>       Raw RGB24 output height after display scaling.
  --upscale <n>             Integer output scale factor. Cannot be combined with output size.
  --scale-mode <nearest|linear>
                             Display scaling mode. Default: nearest.
  --control-port <n>        Listen for realtime UDP control messages on 127.0.0.1:n.
  --block-size <n>          Motion block size. Default: 16
  --search-radius <n>       Motion search radius in pixels. Default: 8
  --search-step <n>         Motion search step in pixels. Default: 4
  --keyframe-every <n>      Emit an internal I-frame every nth input frame. 0 means first only.
  --history <n>             Reference frame history length. Default: 8
  --reference-mode <split|feedback>
                             split keeps encoder input history clean; feedback re-encodes glitches.

Glitch options:
  --preset <clean|subtle|classic|melt|grain|pixel|residue|scan|drift|bank|plane|vector|entropy|coeff|codebook|unstable|balanced|destroy>
                             Load a raw-mosh parameter preset. Later options can override it.
  --mv-scale <n>            Scale both motion-vector axes.
  --mv-scale-x <n>          Scale horizontal motion vectors. Default: 1.0
  --mv-scale-y <n>          Scale vertical motion vectors. Default: 1.0
  --mv-jitter <n>           Deterministic signed jitter added to motion vectors.
  --mv-quant <n>            Quantize motion vectors to this pixel grid. Default: 1
  --reference-lag <n>       Decode from the nth previous reconstructed frame. Default: 1
  --residual-keep <n>       Residual multiplier. 1 reconstructs, 0 pure motion smear.
  --residual-invert-every <n>
                             Invert residual on every nth block. 0 disables.
  --residual-address-shift-x <n>
                             Read residual samples from a horizontally shifted address.
  --residual-address-shift-y <n>
                             Read residual samples from a vertically shifted address.
  --residual-address-jitter <n>
                             Add deterministic residual address jitter up to n pixels.
  --residual-channel-shift <n>
                             Rotate residual channel reads.
  --temporal-slice-height <n>
                             Horizontal stripe height for temporal slice drift. 0 disables.
  --temporal-slice-lag-span <n>
                             Reference history span used by temporal slice drift. 0 disables.
  --temporal-slice-drift <n>
                             Signed lag drift applied each latch bucket.
  --residual-bank-size <n>
                             Residual cell size for residual-bank misreads. 0 disables.
  --residual-bank-stride <n>
                             Signed residual-bank offset used when reading residuals.
  --residual-bank-shuffle-every <n>
                             Randomize selected residual-bank offsets every nth bank. 0 disables.
  --reference-channel-shift <n>
                             Rotate reference channel reads.
  --reference-channel-lag-span <n>
                             Reference history span for per-channel plane desync. 0 disables.
  --reference-channel-lag-stride <n>
                             Signed per-channel reference lag offset.
  --mv-bank-size <n>        Motion-vector bank cell size in blocks. 0 disables.
  --mv-bank-stride <n>      Signed motion-vector bank offset.
  --mv-bank-shuffle-every <n>
                             Randomize selected motion-vector bank offsets every nth bank. 0 disables.
  --block-remap-every <n>   Use another block's motion vector every nth block. 0 disables.
  --block-remap-stride <n>  Block-vector offset used by remap.
  --channel-shift <n>       Offset G/B channel reference sampling horizontally.
  --activity-mode <all|active|static>
                             Apply glitch to all blocks, active/different blocks, or static blocks.
  --activity-threshold <n>   Activity threshold for active/static block gating. Default: 12
  --activity-softness <n>    Soft transition range above the activity threshold. 0 is binary.
  --reference-bleed <n>      Minimum hard-switch chance to dirty references, 0.0-1.0.
  --reference-latch-frames <n>
                             Keep dirty-reference switch decisions stable for this many frames.
  --reference-slots <n>      Dirty decoded reference slots available for sample-level misreads.
  --reference-slot-shuffle-every <n>
                             Use a wrong dirty reference slot on distributed sample cells.
  --reference-scanline-height <n>
                             Stripe height for scanline reference-history desync. 0 disables.
  --reference-scanline-lag-span <n>
                             Reference history span used by scanline desync. 0 disables.
  --overlap <n>              Overlapped block-compensation radius in pixels. 0 disables.
  --motion-diffusion <n>     Blend each motion vector toward neighbor vectors, 0.0-1.0.
  --mv-field-interpolation <n>
                             Interpolate decoded motion vectors per pixel, 0.0-1.0.
  --sample-address-desync <n>
                             Corrupt dirty reference sample addresses by up to n pixels.
  --pixel-grain <n>          Pixel-cell size for sample-level glitches. 0 uses tuned defaults;
                             1 hashes every pixel independently.
  --pixel-grain-x <n>        Horizontal pixel-cell size. Overrides --pixel-grain on X.
  --pixel-grain-y <n>        Vertical pixel-cell size. Overrides --pixel-grain on Y.
  --mv-predictor-desync-every <n>
                             Desync the motion-vector predictor on every nth block.
  --mv-predictor-desync-x <n>
                             Horizontal predictor desync delta.
  --mv-predictor-desync-y <n>
                             Vertical predictor desync delta.
  --wrap-motion             Wrap out-of-range motion sampling for harsher smears.
  --clamp-motion            Clamp out-of-range motion sampling. Default.
  --bitstream               Serialize to MSH0 packet bytes and decode through the bitstream path.
  --bs-mv-sign-flip-every <n>
                             Flip serialized motion-vector signs every nth block.
  --bs-mv-delta-every <n>   Add a serialized motion-vector delta every nth block.
  --bs-mv-delta-x <n>       Horizontal delta for --bs-mv-delta-every.
  --bs-mv-delta-y <n>       Vertical delta for --bs-mv-delta-every.
  --bs-block-shift-every <n>
                             Shift serialized block destination addresses every nth block.
  --bs-block-shift-x <n>    Horizontal block address shift.
  --bs-block-shift-y <n>    Vertical block address shift.
  --bs-residual-zero-every <n>
                             Zero serialized residual samples every nth block.
  --bs-residual-xor-every <n>
                             XOR serialized residual bytes every nth block.
  --bs-residual-xor-mask <n>
                             Byte mask for residual XOR. Default: 255
  --bs-entropy-slip-every <n>
                             Byte-slip residual payload windows every nth P frame. 0 disables.
  --bs-entropy-slip-by <n>  Signed byte rotation for entropy slip windows. Default: 1
  --bs-entropy-resync-bytes <n>
                             Residual payload window length before simulated resync. 0 means whole payload.
  --bs-entropy-windows <n>  Number of entropy slip windows per affected frame. Default: 1
  --bs-coeff-every <n>      Transform residual payload tiles every nth P frame. 0 disables.
  --bs-coeff-block-size <n> Hadamard residual transform tile size: 4, 8, or 16. Default: 8
  --bs-coeff-shift <n>      Signed coefficient rotation excluding DC. 0 disables.
  --bs-coeff-sign-flip-every <n>
                             Flip every nth transformed residual coefficient. 0 disables.
  --bs-coeff-zero-high <n>  Zero coefficients where x+y is at least n. 0 disables.
  --bs-coeff-quant <n>      Quantize transformed residual coefficients. 1 disables.
  --bs-codebook-every <n>   Replace every nth residual tile with a codebook tile. 0 disables.
  --bs-codebook-tile-size <n>
                             Residual codebook tile size: 4, 8, or 16. Default: 8
  --bs-codebook-slots <n>   Maximum residual tiles kept in the decoder codebook.
  --bs-codebook-stride <n>  Signed codebook slot offset used for replacement.
  --bs-codebook-update-every <n>
                             Store every nth clean residual tile in the codebook.
  --bs-codebook-shuffle-every <n>
                             Randomize selected codebook slot reads. 0 disables.
  --quiet                   Do not print stats.
  -h, --help                Show this help.

Realtime UDP control messages:
  preset <name>             Load a raw-mosh preset without restarting the pipe.
  controls <i> <m> <r> <t> <b>
                             Set normalized intensity, motion, residual, temporal, bitstream.
  set <id> <value>          Set a named raw-mosh parameter.
  reset-glitch              Clear dirty reference history and residual codebook state.
"
    );
}


fn run_cli_stream(
    cli: CliArgs,
    mut input: impl Read,
    mut output: impl Write,
    mut err: impl Write,
) -> io::Result<()> {
    let mut stream = DatamoshStream::new(cli.config);
    if let Some(path) = cli.donor_file {
        let mut donor = File::open(&path)?;
        load_donor_stream(&mut stream, &mut donor)?;
    }

    run_stream_inner(&mut stream, &mut input, &mut output, &mut err)
}


fn bind_raw_mosh_control_socket(port: Option<u16>) -> io::Result<Option<UdpSocket>> {
    let Some(port) = port else {
        return Ok(None);
    };
    let socket = UdpSocket::bind(("127.0.0.1", port))?;
    socket.set_nonblocking(true)?;
    Ok(Some(socket))
}


fn drain_raw_mosh_control_socket(
    socket: Option<&UdpSocket>,
    config: &mut MoshCodecConfig,
    params: &mut MoshGlitchParams,
    bitstream: &mut MoshBitstreamParams,
    controls: &mut RawMoshControls,
    err: &mut dyn Write,
    quiet: bool,
) -> io::Result<RawMoshControlUpdate> {
    let Some(socket) = socket else {
        return Ok(RawMoshControlUpdate::default());
    };

    let mut update = RawMoshControlUpdate::default();
    let mut packet = [0_u8; 2048];

    loop {
        match socket.recv_from(&mut packet) {
            Ok((len, _addr)) => {
                let message = String::from_utf8_lossy(&packet[..len]);
                match apply_raw_mosh_control_message(&message, config, params, bitstream, controls)
                {
                    Ok(control_update) => {
                        update.rebuild_codec |= control_update.rebuild_codec;
                        update.reset_glitch_state |= control_update.reset_glitch_state;
                    }
                    Err(message) => {
                        if !quiet {
                            writeln!(err, "datamosh raw-mosh control: {message}")?;
                        }
                    }
                }
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }

    Ok(update)
}


fn run_raw_mosh_stream(
    cli: RawMoshCli,
    mut input: impl Read,
    mut output: impl Write,
    mut err: impl Write,
) -> io::Result<()> {
    let RawMoshCli {
        mut config,
        mut params,
        mut bitstream,
        output_width,
        output_height,
        scale_mode,
        control_port,
        quiet,
    } = cli;
    let frame_len = config.frame_len().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "raw-mosh frame dimensions overflow addressable memory",
        )
    })?;
    let input_width = config.width;
    let input_height = config.height;
    let output_width = output_width.unwrap_or(input_width);
    let output_height = output_height.unwrap_or(input_height);
    let output_frame_len = raw_rgb_frame_len(output_width, output_height)?;
    let scale_output = output_width != input_width || output_height != input_height;
    let control_socket = bind_raw_mosh_control_socket(control_port)?;
    let mut codec = MoshCodec::new(config.clone())?;
    let mut controls = RawMoshControls::default();
    let mut bitstream_stats = MoshBitstreamMutationStats::default();
    let mut input_frame = vec![0_u8; frame_len];
    let mut output_frame = vec![0_u8; frame_len];
    let mut scaled_output_frame = if scale_output {
        vec![0_u8; output_frame_len]
    } else {
        Vec::new()
    };
    let mut last_report = Instant::now();

    loop {
        let mut read_total = 0;
        while read_total < frame_len {
            let read = input.read(&mut input_frame[read_total..])?;
            if read == 0 {
                if read_total == 0 {
                    if let Err(err) = output.flush() {
                        if err.kind() == io::ErrorKind::BrokenPipe {
                            return Ok(());
                        }
                        return Err(err);
                    }
                    if !quiet {
                        write_raw_mosh_report(&codec, &bitstream_stats, &mut err, true)?;
                    }
                    return Ok(());
                }
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "partial raw RGB24 frame at end of input",
                ));
            }
            read_total += read;
        }

        let control_update = drain_raw_mosh_control_socket(
            control_socket.as_ref(),
            &mut config,
            &mut params,
            &mut bitstream,
            &mut controls,
            &mut err,
            quiet,
        )?;
        if control_update.rebuild_codec {
            codec = MoshCodec::new(config.clone())?;
        } else if control_update.reset_glitch_state {
            codec.reset_glitch_state();
        }

        let mut frame_params = params.clone();
        let mut frame_bitstream = bitstream.clone();
        apply_raw_mosh_controls(&mut frame_params, &mut frame_bitstream, controls);

        if frame_bitstream.enabled || frame_bitstream.has_mutations() {
            let stats = codec.process_rgb_frame_bitstream(
                &input_frame,
                &frame_params,
                &frame_bitstream,
                &mut output_frame,
            )?;
            bitstream_stats.mv_sign_flipped += stats.mv_sign_flipped;
            bitstream_stats.mv_delta_applied += stats.mv_delta_applied;
            bitstream_stats.block_addresses_shifted += stats.block_addresses_shifted;
            bitstream_stats.residual_blocks_zeroed += stats.residual_blocks_zeroed;
            bitstream_stats.residual_blocks_xored += stats.residual_blocks_xored;
            bitstream_stats.entropy_slips += stats.entropy_slips;
            bitstream_stats.coeff_blocks += stats.coeff_blocks;
            bitstream_stats.codebook_tiles += stats.codebook_tiles;
        } else {
            codec.process_rgb_frame(&input_frame, &frame_params, &mut output_frame)?;
        }
        let frame_to_write = if scale_output {
            scale_rgb24_frame(
                &output_frame,
                input_width,
                input_height,
                output_width,
                output_height,
                scale_mode,
                &mut scaled_output_frame,
            )?;
            scaled_output_frame.as_slice()
        } else {
            output_frame.as_slice()
        };
        if let Err(err) = output.write_all(frame_to_write) {
            if err.kind() == io::ErrorKind::BrokenPipe {
                return Ok(());
            }
            return Err(err);
        }

        if !quiet && last_report.elapsed() >= REPORT_INTERVAL {
            write_raw_mosh_report(&codec, &bitstream_stats, &mut err, false)?;
            last_report = Instant::now();
        }
    }
}


fn write_raw_mosh_report(
    codec: &MoshCodec,
    bitstream_stats: &MoshBitstreamMutationStats,
    err: &mut dyn Write,
    final_report: bool,
) -> io::Result<()> {
    let prefix = if final_report {
        "datamosh raw-mosh: final"
    } else {
        "datamosh raw-mosh"
    };
    let stats = codec.stats();
    writeln!(
        err,
        "{prefix}: frames {} key/predicted {}/{} blocks {}  bitstream mv_flip/mv_delta/block_shift/residual_zero/residual_xor/entropy_slip/coeff/codebook {}/{}/{}/{}/{}/{}/{}/{}",
        stats.frames_in,
        stats.keyframes,
        stats.predicted_frames,
        stats.blocks_encoded,
        bitstream_stats.mv_sign_flipped,
        bitstream_stats.mv_delta_applied,
        bitstream_stats.block_addresses_shifted,
        bitstream_stats.residual_blocks_zeroed,
        bitstream_stats.residual_blocks_xored,
        bitstream_stats.entropy_slips,
        bitstream_stats.coeff_blocks,
        bitstream_stats.codebook_tiles
    )
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_donor_cli_options() {
        let cli = parse_args(
            [
                "filter",
                "--donor-file",
                "donor.h264",
                "--donor-splice-slice-every",
                "3",
                "--donor-bank-size=4",
            ]
            .into_iter()
            .map(str::to_string),
        )
        .unwrap()
        .unwrap();

        assert_eq!(cli.donor_file.as_deref(), Some("donor.h264"));
        assert_eq!(cli.config.donor_splice_slice_every, 3);
        assert_eq!(cli.config.donor_bank_size, 4);
    }

    #[test]
    fn parses_raw_mosh_cli_options() {
        let cli = parse_raw_mosh_args(
            [
                "--width",
                "64",
                "--height=48",
                "--control-port",
                "24000",
                "--output-width",
                "128",
                "--output-height=96",
                "--scale-mode",
                "linear",
                "--residual-keep",
                "0.25",
                "--residual-address-shift-x=2",
                "--residual-address-shift-y",
                "-1",
                "--residual-address-jitter",
                "5",
                "--residual-channel-shift=1",
                "--temporal-slice-height=6",
                "--temporal-slice-lag-span",
                "8",
                "--temporal-slice-drift=-1",
                "--residual-bank-size",
                "12",
                "--residual-bank-stride=-3",
                "--residual-bank-shuffle-every",
                "5",
                "--reference-channel-shift=1",
                "--reference-channel-lag-span",
                "7",
                "--reference-channel-lag-stride=-2",
                "--mv-bank-size=2",
                "--mv-bank-stride",
                "4",
                "--mv-bank-shuffle-every=6",
                "--mv-scale=2.0",
                "--reference-mode=feedback",
                "--block-remap-every",
                "3",
                "--block-remap-stride=-2",
                "--activity-mode=active",
                "--activity-threshold",
                "15",
                "--activity-softness=20",
                "--reference-bleed",
                "0.3",
                "--reference-latch-frames",
                "7",
                "--reference-slots",
                "5",
                "--reference-slot-shuffle-every",
                "4",
                "--reference-scanline-height=3",
                "--reference-scanline-lag-span",
                "9",
                "--overlap",
                "6",
                "--motion-diffusion=0.4",
                "--mv-field-interp",
                "0.75",
                "--sample-address-desync=1.25",
                "--pixel-grain",
                "1",
                "--pixel-grain-x=4",
                "--pixel-grain-y",
                "2",
                "--mv-predictor-desync-every",
                "6",
                "--mv-predictor-desync-x=2",
                "--mv-predictor-desync-y",
                "-1",
                "--wrap-motion",
            ]
            .into_iter()
            .map(str::to_string),
        )
        .unwrap()
        .unwrap();

        assert_eq!(cli.config.width, 64);
        assert_eq!(cli.config.height, 48);
        assert_eq!(cli.control_port, Some(24000));
        assert_eq!(cli.output_width, Some(128));
        assert_eq!(cli.output_height, Some(96));
        assert_eq!(cli.scale_mode, RawMoshScaleMode::Linear);
        assert_eq!(cli.params.residual_keep, 0.25);
        assert_eq!(cli.params.residual_address_shift_x, 2);
        assert_eq!(cli.params.residual_address_shift_y, -1);
        assert_eq!(cli.params.residual_address_jitter, 5);
        assert_eq!(cli.params.residual_channel_shift, 1);
        assert_eq!(cli.params.temporal_slice_height, 6);
        assert_eq!(cli.params.temporal_slice_lag_span, 8);
        assert_eq!(cli.params.temporal_slice_drift, -1);
        assert_eq!(cli.params.residual_bank_size, 12);
        assert_eq!(cli.params.residual_bank_stride, -3);
        assert_eq!(cli.params.residual_bank_shuffle_every, 5);
        assert_eq!(cli.params.reference_channel_shift, 1);
        assert_eq!(cli.params.reference_channel_lag_span, 7);
        assert_eq!(cli.params.reference_channel_lag_stride, -2);
        assert_eq!(cli.params.mv_bank_size, 2);
        assert_eq!(cli.params.mv_bank_stride, 4);
        assert_eq!(cli.params.mv_bank_shuffle_every, 6);
        assert_eq!(cli.config.reference_mode, MoshReferenceMode::Feedback);
        assert_eq!(cli.params.mv_scale_x, 2.0);
        assert_eq!(cli.params.mv_scale_y, 2.0);
        assert_eq!(cli.params.block_remap_every, 3);
        assert_eq!(cli.params.block_remap_stride, -2);
        assert_eq!(cli.params.activity_mode, ActivityMode::Active);
        assert_eq!(cli.params.activity_threshold, 15);
        assert_eq!(cli.params.activity_softness, 20);
        assert_eq!(cli.params.reference_bleed, 0.3);
        assert_eq!(cli.params.reference_latch_frames, 7);
        assert_eq!(cli.params.reference_slot_count, 5);
        assert_eq!(cli.params.reference_slot_shuffle_every, 4);
        assert_eq!(cli.params.reference_scanline_height, 3);
        assert_eq!(cli.params.reference_scanline_lag_span, 9);
        assert_eq!(cli.params.overlap, 6);
        assert_eq!(cli.params.motion_diffusion, 0.4);
        assert_eq!(cli.params.mv_field_interpolation, 0.75);
        assert_eq!(cli.params.sample_address_desync, 1.25);
        assert_eq!(cli.params.glitch_cell_size, 1);
        assert_eq!(cli.params.glitch_cell_width, 4);
        assert_eq!(cli.params.glitch_cell_height, 2);
        assert_eq!(cli.params.mv_predictor_desync_every, 6);
        assert_eq!(cli.params.mv_predictor_desync_x, 2);
        assert_eq!(cli.params.mv_predictor_desync_y, -1);
        assert!(cli.params.wrap_motion);
    }

    #[test]
    fn raw_mosh_preset_can_be_overridden_by_later_options() {
        let cli = parse_raw_mosh_args(
            [
                "--width",
                "64",
                "--height",
                "48",
                "--preset",
                "subtle",
                "--residual-keep",
                "0.9",
                "--wrap-motion",
            ]
            .into_iter()
            .map(str::to_string),
        )
        .unwrap()
        .unwrap();

        assert_eq!(cli.params.mv_jitter, 1);
        assert_eq!(cli.params.reference_lag, 2);
        assert_eq!(cli.params.residual_keep, 0.9);
        assert!(cli.params.wrap_motion);
    }

    #[test]
    fn raw_mosh_clean_preset_is_neutral() {
        let cli = parse_raw_mosh_args(
            ["--width=64", "--height=48", "--preset=clean"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap()
        .unwrap();

        assert_eq!(cli.config.block_size, 16);
        assert_eq!(cli.config.search_radius, 8);
        assert_eq!(cli.config.search_step, 4);
        assert_eq!(cli.params.reference_lag, 1);
        assert_eq!(cli.params.residual_keep, 1.0);
        assert_eq!(cli.params.mv_jitter, 0);
        assert_eq!(cli.params.residual_address_jitter, 0);
        assert!(!cli.bitstream.enabled);
        assert!(!cli.bitstream.has_mutations());
    }

    #[test]
    fn raw_mosh_classic_preset_keeps_residual_and_plain_motion() {
        let cli = parse_raw_mosh_args(
            ["--width=64", "--height=48", "--preset=classic"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap()
        .unwrap();

        assert_eq!(cli.params.mv_scale_x, 1.0);
        assert_eq!(cli.params.mv_scale_y, 1.0);
        assert_eq!(cli.params.reference_lag, 3);
        assert_eq!(cli.params.residual_keep, 0.22);
        assert_eq!(cli.params.block_remap_every, 0);
        assert_eq!(cli.params.activity_mode, ActivityMode::Active);
        assert_eq!(cli.params.activity_threshold, 12);
        assert_eq!(cli.params.activity_softness, 0);
        assert_eq!(cli.params.reference_bleed, 0.12);
        assert_eq!(cli.params.reference_latch_frames, 5);
        assert_eq!(cli.params.reference_slot_count, 4);
        assert_eq!(cli.params.reference_slot_shuffle_every, 11);
        assert_eq!(cli.params.overlap, 0);
        assert_eq!(cli.params.motion_diffusion, 0.0);
        assert_eq!(cli.params.mv_field_interpolation, 0.7);
        assert_eq!(cli.params.sample_address_desync, 0.65);
        assert_eq!(cli.params.glitch_cell_size, 0);
        assert_eq!(cli.params.glitch_cell_width, 0);
        assert_eq!(cli.params.glitch_cell_height, 0);
        assert_eq!(cli.params.mv_predictor_desync_every, 17);
        assert_eq!(cli.params.mv_predictor_desync_x, 1);
        assert_eq!(cli.params.mv_predictor_desync_y, 0);
        assert!(!cli.params.wrap_motion);
    }

    #[test]
    fn raw_mosh_melt_preset_targets_active_blocks_with_longer_dirty_reference() {
        let cli = parse_raw_mosh_args(
            ["--width=64", "--height=48", "--preset=melt"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap()
        .unwrap();

        assert_eq!(cli.config.reference_mode, MoshReferenceMode::Split);
        assert_eq!(cli.params.mv_jitter, 1);
        assert_eq!(cli.params.mv_quant, 2);
        assert_eq!(cli.params.reference_lag, 8);
        assert_eq!(cli.params.residual_keep, 0.0);
        assert_eq!(cli.params.activity_mode, ActivityMode::Active);
        assert_eq!(cli.params.activity_threshold, 10);
        assert_eq!(cli.params.activity_softness, 0);
        assert_eq!(cli.params.reference_bleed, 0.16);
        assert_eq!(cli.params.reference_latch_frames, 6);
        assert_eq!(cli.params.reference_slot_count, 6);
        assert_eq!(cli.params.reference_slot_shuffle_every, 9);
        assert_eq!(cli.params.overlap, 0);
        assert_eq!(cli.params.motion_diffusion, 0.0);
        assert_eq!(cli.params.mv_field_interpolation, 0.85);
        assert_eq!(cli.params.sample_address_desync, 1.15);
        assert_eq!(cli.params.glitch_cell_size, 0);
        assert_eq!(cli.params.glitch_cell_width, 0);
        assert_eq!(cli.params.glitch_cell_height, 0);
        assert_eq!(cli.params.mv_predictor_desync_every, 11);
        assert_eq!(cli.params.mv_predictor_desync_x, 2);
        assert_eq!(cli.params.mv_predictor_desync_y, -1);
        assert_eq!(cli.params.block_remap_every, 0);
    }

    #[test]
    fn raw_mosh_grain_preset_uses_medium_glitch_grain() {
        let cli = parse_raw_mosh_args(
            ["--width=64", "--height=48", "--preset=grain"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap()
        .unwrap();

        assert_eq!(cli.params.reference_lag, 8);
        assert_eq!(cli.params.reference_slot_count, 7);
        assert_eq!(cli.params.reference_slot_shuffle_every, 6);
        assert_eq!(cli.params.mv_quant, 1);
        assert_eq!(cli.params.mv_field_interpolation, 0.88);
        assert_eq!(cli.params.sample_address_desync, 1.3);
        assert_eq!(cli.params.glitch_cell_size, 0);
        assert_eq!(cli.params.glitch_cell_width, 3);
        assert_eq!(cli.params.glitch_cell_height, 2);
        assert_eq!(cli.params.mv_predictor_desync_every, 10);
    }

    #[test]
    fn raw_mosh_pixel_preset_uses_one_pixel_glitch_grain() {
        let cli = parse_raw_mosh_args(
            ["--width=64", "--height=48", "--preset=pixel"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap()
        .unwrap();

        assert_eq!(cli.params.reference_lag, 8);
        assert_eq!(cli.params.reference_slot_count, 8);
        assert_eq!(cli.params.reference_slot_shuffle_every, 5);
        assert_eq!(cli.params.mv_quant, 1);
        assert_eq!(cli.params.mv_field_interpolation, 0.9);
        assert_eq!(cli.params.sample_address_desync, 1.25);
        assert_eq!(cli.params.glitch_cell_size, 1);
        assert_eq!(cli.params.glitch_cell_width, 1);
        assert_eq!(cli.params.glitch_cell_height, 1);
        assert_eq!(cli.params.mv_predictor_desync_every, 9);
    }

    #[test]
    fn raw_mosh_residue_preset_corrupts_residual_stream() {
        let cli = parse_raw_mosh_args(
            ["--width=64", "--height=48", "--preset=residue"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap()
        .unwrap();

        assert_eq!(cli.params.reference_lag, 1);
        assert_eq!(cli.config.block_size, 32);
        assert_eq!(cli.config.search_radius, 0);
        assert_eq!(cli.config.search_step, 1);
        assert_eq!(cli.params.residual_keep, 1.35);
        assert_eq!(cli.params.residual_address_shift_x, 2);
        assert_eq!(cli.params.residual_address_jitter, 5);
        assert_eq!(cli.params.residual_channel_shift, 1);
        assert_eq!(cli.params.activity_mode, ActivityMode::All);
        assert_eq!(cli.params.reference_scanline_height, 0);
    }

    #[test]
    fn raw_mosh_scan_preset_corrupts_reference_history_by_scanline() {
        let cli = parse_raw_mosh_args(
            ["--width=64", "--height=48", "--preset=scan"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap()
        .unwrap();

        assert_eq!(cli.params.reference_lag, 1);
        assert_eq!(cli.config.block_size, 32);
        assert_eq!(cli.config.search_radius, 0);
        assert_eq!(cli.config.search_step, 1);
        assert_eq!(cli.params.residual_keep, 0.65);
        assert_eq!(cli.params.reference_latch_frames, 2);
        assert_eq!(cli.params.reference_scanline_height, 2);
        assert_eq!(cli.params.reference_scanline_lag_span, 4);
        assert_eq!(cli.params.residual_address_jitter, 0);
        assert_eq!(cli.params.activity_mode, ActivityMode::All);
    }

    #[test]
    fn raw_mosh_drift_preset_uses_temporal_slice_history_drift() {
        let cli = parse_raw_mosh_args(
            ["--width=64", "--height=48", "--preset=drift"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap()
        .unwrap();

        assert_eq!(cli.config.block_size, 32);
        assert_eq!(cli.config.search_radius, 0);
        assert_eq!(cli.config.search_step, 1);
        assert_eq!(cli.params.reference_lag, 1);
        assert_eq!(cli.params.residual_keep, 0.75);
        assert_eq!(cli.params.reference_latch_frames, 2);
        assert_eq!(cli.params.temporal_slice_height, 12);
        assert_eq!(cli.params.temporal_slice_lag_span, 8);
        assert_eq!(cli.params.temporal_slice_drift, 1);
        assert_eq!(cli.params.reference_scanline_height, 0);
        assert_eq!(cli.params.residual_bank_size, 0);
    }

    #[test]
    fn raw_mosh_bank_preset_corrupts_residual_banks() {
        let cli = parse_raw_mosh_args(
            ["--width=64", "--height=48", "--preset=bank"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap()
        .unwrap();

        assert_eq!(cli.config.block_size, 32);
        assert_eq!(cli.config.search_radius, 0);
        assert_eq!(cli.config.search_step, 1);
        assert_eq!(cli.params.reference_lag, 1);
        assert_eq!(cli.params.residual_keep, 1.2);
        assert_eq!(cli.params.residual_channel_shift, 1);
        assert_eq!(cli.params.residual_bank_size, 24);
        assert_eq!(cli.params.residual_bank_stride, 5);
        assert_eq!(cli.params.residual_bank_shuffle_every, 3);
        assert_eq!(cli.params.temporal_slice_height, 0);
        assert_eq!(cli.params.activity_mode, ActivityMode::All);
    }

    #[test]
    fn raw_mosh_plane_preset_desyncs_reference_channels() {
        let cli = parse_raw_mosh_args(
            ["--width=64", "--height=48", "--preset=plane"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap()
        .unwrap();

        assert_eq!(cli.config.block_size, 32);
        assert_eq!(cli.config.search_radius, 0);
        assert_eq!(cli.config.search_step, 1);
        assert_eq!(cli.params.residual_keep, 0.85);
        assert_eq!(cli.params.residual_channel_shift, -1);
        assert_eq!(cli.params.reference_channel_shift, 1);
        assert_eq!(cli.params.reference_channel_lag_span, 6);
        assert_eq!(cli.params.reference_channel_lag_stride, 1);
        assert_eq!(cli.params.mv_bank_size, 0);
        assert_eq!(cli.params.activity_mode, ActivityMode::All);
    }

    #[test]
    fn raw_mosh_vector_preset_corrupts_motion_vector_banks() {
        let cli = parse_raw_mosh_args(
            ["--width=64", "--height=48", "--preset=vector"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap()
        .unwrap();

        assert_eq!(cli.config.block_size, 24);
        assert_eq!(cli.config.search_radius, 8);
        assert_eq!(cli.config.search_step, 4);
        assert_eq!(cli.params.reference_lag, 5);
        assert_eq!(cli.params.residual_keep, 0.16);
        assert_eq!(cli.params.mv_bank_size, 2);
        assert_eq!(cli.params.mv_bank_stride, 3);
        assert_eq!(cli.params.mv_bank_shuffle_every, 4);
        assert_eq!(cli.params.mv_field_interpolation, 0.82);
        assert_eq!(cli.params.activity_mode, ActivityMode::Active);
    }

    #[test]
    fn raw_mosh_entropy_preset_uses_bitstream_byte_slip() {
        let cli = parse_raw_mosh_args(
            ["--width=64", "--height=48", "--preset=entropy"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap()
        .unwrap();

        assert_eq!(cli.config.block_size, 16);
        assert_eq!(cli.config.search_radius, 4);
        assert_eq!(cli.config.search_step, 4);
        assert_eq!(cli.params.reference_lag, 1);
        assert_eq!(cli.params.residual_keep, 1.0);
        assert!(cli.bitstream.enabled);
        assert_eq!(cli.bitstream.entropy_slip_every, 2);
        assert_eq!(cli.bitstream.entropy_slip_bytes, 1);
        assert_eq!(cli.bitstream.entropy_resync_bytes, 4096);
        assert_eq!(cli.bitstream.entropy_slip_windows, 4);
    }

    #[test]
    fn raw_mosh_coeff_preset_uses_transform_coefficients() {
        let cli = parse_raw_mosh_args(
            ["--width=64", "--height=48", "--preset=coeff"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap()
        .unwrap();

        assert_eq!(cli.config.block_size, 16);
        assert_eq!(cli.config.search_radius, 4);
        assert_eq!(cli.config.search_step, 4);
        assert_eq!(cli.params.reference_lag, 1);
        assert_eq!(cli.params.residual_keep, 1.0);
        assert!(cli.bitstream.enabled);
        assert_eq!(cli.bitstream.coeff_glitch_every, 1);
        assert_eq!(cli.bitstream.coeff_block_size, 8);
        assert_eq!(cli.bitstream.coeff_shift, 2);
        assert_eq!(cli.bitstream.coeff_sign_flip_every, 19);
        assert_eq!(cli.bitstream.coeff_zero_high, 12);
        assert_eq!(cli.bitstream.coeff_quant, 4);
    }

    #[test]
    fn raw_mosh_codebook_preset_uses_residual_dictionary() {
        let cli = parse_raw_mosh_args(
            ["--width=64", "--height=48", "--preset=codebook"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap()
        .unwrap();

        assert_eq!(cli.config.block_size, 16);
        assert_eq!(cli.config.search_radius, 4);
        assert_eq!(cli.config.search_step, 4);
        assert_eq!(cli.params.reference_lag, 1);
        assert_eq!(cli.params.residual_keep, 1.0);
        assert!(cli.bitstream.enabled);
        assert_eq!(cli.bitstream.codebook_replace_every, 9);
        assert_eq!(cli.bitstream.codebook_tile_size, 8);
        assert_eq!(cli.bitstream.codebook_slots, 96);
        assert_eq!(cli.bitstream.codebook_stride, -17);
        assert_eq!(cli.bitstream.codebook_update_every, 3);
        assert_eq!(cli.bitstream.codebook_shuffle_every, 5);
    }

    #[test]
    fn raw_mosh_realtime_control_messages_update_runtime_state() {
        let mut config = MoshCodecConfig::new(64, 48);
        let mut params = MoshGlitchParams::default();
        let mut bitstream = MoshBitstreamParams::default();
        let mut controls = RawMoshControls::default();

        let update = apply_raw_mosh_control_message(
            "set mv_scale 1.5\nresidual_keep=0.25\ncontrols 0.5 0.75 1.0 0.25 0.0",
            &mut config,
            &mut params,
            &mut bitstream,
            &mut controls,
        )
        .unwrap();

        assert!(!update.rebuild_codec);
        assert_eq!(params.mv_scale_x, 1.5);
        assert_eq!(params.mv_scale_y, 1.5);
        assert_eq!(params.residual_keep, 0.25);
        assert_eq!(controls.intensity, 0.5);
        assert_eq!(controls.motion, 0.75);
        assert_eq!(controls.temporal, 0.25);
        assert_eq!(controls.bitstream, 0.0);

        let update = apply_raw_mosh_control_message(
            "set block_size 24",
            &mut config,
            &mut params,
            &mut bitstream,
            &mut controls,
        )
        .unwrap();

        assert!(update.rebuild_codec);
        assert_eq!(config.block_size, 24);
    }

    #[test]
    fn raw_mosh_realtime_control_reset_requests_glitch_state_clear() {
        let mut config = MoshCodecConfig::new(64, 48);
        let mut params = MoshGlitchParams::default();
        let mut bitstream = MoshBitstreamParams::default();
        let mut controls = RawMoshControls::default();

        let update = apply_raw_mosh_control_message(
            "reset-glitch",
            &mut config,
            &mut params,
            &mut bitstream,
            &mut controls,
        )
        .unwrap();

        assert!(!update.rebuild_codec);
        assert!(update.reset_glitch_state);
    }

    #[test]
    fn raw_mosh_realtime_control_preset_rebuilds_codec_state() {
        let mut config = MoshCodecConfig::new(64, 48);
        let mut params = MoshGlitchParams::default();
        let mut bitstream = MoshBitstreamParams::default();
        let mut controls = RawMoshControls::default();

        let update = apply_raw_mosh_control_message(
            "preset codebook",
            &mut config,
            &mut params,
            &mut bitstream,
            &mut controls,
        )
        .unwrap();

        assert!(update.rebuild_codec);
        assert_eq!(config.block_size, 16);
        assert_eq!(params.residual_keep, 1.0);
        assert!(bitstream.enabled);
        assert_eq!(bitstream.codebook_replace_every, 9);
    }

    #[test]
    fn raw_mosh_upscale_sets_output_dimensions() {
        let cli = parse_raw_mosh_args(
            [
                "--width=64",
                "--height=48",
                "--upscale",
                "3",
                "--scale-mode=nearest",
            ]
            .into_iter()
            .map(str::to_string),
        )
        .unwrap()
        .unwrap();

        assert_eq!(cli.output_width, Some(192));
        assert_eq!(cli.output_height, Some(144));
        assert_eq!(cli.scale_mode, RawMoshScaleMode::Nearest);
    }

    #[test]
    fn raw_mosh_rejects_ambiguous_output_scaling() {
        let err = parse_raw_mosh_args(
            [
                "--width=64",
                "--height=48",
                "--upscale=2",
                "--output-width=128",
                "--output-height=96",
            ]
            .into_iter()
            .map(str::to_string),
        )
        .unwrap_err();

        assert_eq!(
            err,
            "--upscale cannot be combined with --output-width/--output-height"
        );
    }

    #[test]
    fn scales_raw_rgb24_frame_with_nearest_neighbor() {
        let input = vec![1, 2, 3, 10, 20, 30, 100, 110, 120, 200, 210, 220];
        let mut output = vec![0; 4 * 4 * 3];

        scale_rgb24_frame(&input, 2, 2, 4, 4, RawMoshScaleMode::Nearest, &mut output).unwrap();

        assert_eq!(&output[0..3], &[1, 2, 3]);
        assert_eq!(&output[3..6], &[1, 2, 3]);
        assert_eq!(&output[6..9], &[10, 20, 30]);
        assert_eq!(&output[24..27], &[100, 110, 120]);
        assert_eq!(&output[45..48], &[200, 210, 220]);
    }

    #[test]
    fn raw_mosh_preset_config_can_be_overridden_by_later_options() {
        let cli = parse_raw_mosh_args(
            [
                "--width=64",
                "--height=48",
                "--preset=residue",
                "--block-size=16",
                "--search-radius",
                "4",
                "--search-step=2",
            ]
            .into_iter()
            .map(str::to_string),
        )
        .unwrap()
        .unwrap();

        assert_eq!(cli.config.block_size, 16);
        assert_eq!(cli.config.search_radius, 4);
        assert_eq!(cli.config.search_step, 2);
        assert_eq!(cli.params.residual_address_jitter, 5);
    }

    #[test]
    fn raw_mosh_unstable_preset_enables_codec_state_corruption() {
        let cli = parse_raw_mosh_args(
            ["--width=64", "--height=48", "--preset=unstable"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap()
        .unwrap();

        assert_eq!(cli.params.reference_lag, 10);
        assert_eq!(cli.params.reference_bleed, 0.24);
        assert_eq!(cli.params.reference_latch_frames, 9);
        assert_eq!(cli.params.reference_slot_count, 10);
        assert_eq!(cli.params.reference_slot_shuffle_every, 7);
        assert_eq!(cli.params.mv_field_interpolation, 0.85);
        assert_eq!(cli.params.sample_address_desync, 1.8);
        assert_eq!(cli.params.glitch_cell_size, 0);
        assert_eq!(cli.params.glitch_cell_width, 0);
        assert_eq!(cli.params.glitch_cell_height, 0);
        assert_eq!(cli.params.mv_predictor_desync_every, 7);
        assert_eq!(cli.params.mv_predictor_desync_x, 3);
        assert_eq!(cli.params.mv_predictor_desync_y, -2);
    }

    #[test]
    fn parses_raw_mosh_bitstream_options() {
        let cli = parse_raw_mosh_args(
            [
                "--width=64",
                "--height=48",
                "--bitstream",
                "--bs-mv-sign-flip-every=3",
                "--bs-mv-delta-every",
                "4",
                "--bs-mv-delta-x",
                "2",
                "--bs-block-shift-every",
                "5",
                "--bs-block-shift-x=-1",
                "--bs-residual-zero-every=2",
                "--bs-residual-xor-every",
                "6",
                "--bs-residual-xor-mask",
                "17",
                "--bs-entropy-slip-every=2",
                "--bs-entropy-slip-by",
                "-1",
                "--bs-entropy-resync-bytes",
                "128",
                "--bs-entropy-windows=3",
                "--bs-coeff-every=1",
                "--bs-coeff-block-size",
                "8",
                "--bs-coeff-shift=3",
                "--bs-coeff-sign-flip-every",
                "5",
                "--bs-coeff-zero-high=9",
                "--bs-coeff-quant",
                "6",
                "--bs-codebook-every=7",
                "--bs-codebook-tile-size",
                "8",
                "--bs-codebook-slots=33",
                "--bs-codebook-stride",
                "-5",
                "--bs-codebook-update-every=2",
                "--bs-codebook-shuffle-every",
                "4",
            ]
            .into_iter()
            .map(str::to_string),
        )
        .unwrap()
        .unwrap();

        assert!(cli.bitstream.enabled);
        assert_eq!(cli.bitstream.mv_sign_flip_every, 3);
        assert_eq!(cli.bitstream.mv_delta_every, 4);
        assert_eq!(cli.bitstream.mv_delta_x, 2);
        assert_eq!(cli.bitstream.block_address_shift_every, 5);
        assert_eq!(cli.bitstream.block_address_shift_x, -1);
        assert_eq!(cli.bitstream.residual_zero_every, 2);
        assert_eq!(cli.bitstream.residual_xor_every, 6);
        assert_eq!(cli.bitstream.residual_xor_mask, 17);
        assert_eq!(cli.bitstream.entropy_slip_every, 2);
        assert_eq!(cli.bitstream.entropy_slip_bytes, -1);
        assert_eq!(cli.bitstream.entropy_resync_bytes, 128);
        assert_eq!(cli.bitstream.entropy_slip_windows, 3);
        assert_eq!(cli.bitstream.coeff_glitch_every, 1);
        assert_eq!(cli.bitstream.coeff_block_size, 8);
        assert_eq!(cli.bitstream.coeff_shift, 3);
        assert_eq!(cli.bitstream.coeff_sign_flip_every, 5);
        assert_eq!(cli.bitstream.coeff_zero_high, 9);
        assert_eq!(cli.bitstream.coeff_quant, 6);
        assert_eq!(cli.bitstream.codebook_replace_every, 7);
        assert_eq!(cli.bitstream.codebook_tile_size, 8);
        assert_eq!(cli.bitstream.codebook_slots, 33);
        assert_eq!(cli.bitstream.codebook_stride, -5);
        assert_eq!(cli.bitstream.codebook_update_every, 2);
        assert_eq!(cli.bitstream.codebook_shuffle_every, 4);
    }
}
