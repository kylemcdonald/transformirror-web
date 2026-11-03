# OSC Messages

The live runtime listens for OSC over UDP. By default:

```text
Host: 0.0.0.0
Port: 9000
```

The bind host and port come from `osc_host` and `osc_port` in `live_config.json`.
Display offset and display mode changes are persisted to `live_config.json`.

## Address Format

All root addresses also work with the `/transformirror` namespace. For example, `/prompt` and `/transformirror/prompt` are equivalent.

Aliases are listed separately where supported. Values are parsed by `transformirror_live.py`.

## Messages

| Address | Arguments | Effect |
| --- | --- | --- |
| `/prompt` | `string` | Sets the prompt. |
| `/seed` | `int` | Sets the seed. |
| `/strength` | `float 0..1` | Sets diffusion strength. Values are clamped to `0..1`. |
| `/blend` | `float 0..1` | Sets output blend. `0` is raw camera, `1` is processed output. Values are clamped to `0..1`. |
| `/passthrough` | `bool` | Enables raw camera passthrough when true, processed output when false. True values are `1`, `true`, `yes`, `on`. |
| `/resolution` | `int width, int height` | Sets diffusion resolution. Width and height are clamped to `32..1280` and rounded down to 32-pixel steps. |
| `/resolution` | `string` | Sets diffusion resolution from text such as `1024x512`. |
| `/width` | `int` | Sets diffusion width while keeping the current height. |
| `/height` | `int` | Sets diffusion height while keeping the current width. |
| `/display_offset` | `int x, int y` | Sets display offset from the centered fitted image. Positive `x` moves right; positive `y` moves down. |
| `/display_offset_x` | `int` | Sets display X offset in pixels. Positive values move right. |
| `/display_offset_y` | `int` | Sets display Y offset in pixels. Positive values move down. |
| `/display_mode` | `string` | Sets display mode. Use `fit` for aspect fit or `fill` for aspect fill crop. Aliases accepted by the runtime include `aspect_fit`, `contain`, `aspect_fill`, and `cover`. |
| `/aspect_fill` | `bool` | Sets display mode from a boolean. True values select `fill`; false values select `fit`. |
| `/offset` | `int x, int y` | Alias for `/display_offset`. |
| `/offset_x` | `int` | Alias for `/display_offset_x`. |
| `/offset_y` | `int` | Alias for `/display_offset_y`. |
| `/steps` | `int 2..8` | Sets diffusion inference steps. Values are clamped to `2..8`. |
| `/screenshot` | `string path` | Saves the current composed display frame to `path`. |

## Namespaced Addresses

These namespaced addresses are equivalent to the root addresses above:

```text
/transformirror/prompt
/transformirror/seed
/transformirror/strength
/transformirror/blend
/transformirror/passthrough
/transformirror/resolution
/transformirror/width
/transformirror/height
/transformirror/display_offset
/transformirror/display_offset_x
/transformirror/display_offset_y
/transformirror/display_mode
/transformirror/aspect_fill
/transformirror/offset
/transformirror/offset_x
/transformirror/offset_y
/transformirror/steps
/transformirror/screenshot
```

## Examples

```text
/prompt "a neon mirror portrait"
/seed 42
/strength 0.7
/blend 0.5
/passthrough true
/resolution 1024 512
/resolution "1024x512"
/display_offset 40 -20
/display_mode "fill"
/aspect_fill true
/offset_x 40
/offset_y -20
/steps 2
/screenshot "/tmp/transformirror.jpg"
```
