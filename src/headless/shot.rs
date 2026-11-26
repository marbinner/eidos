//! `eidos shot`: render a scene without a window, with scripted input, and save a PNG.

use super::{Options, Res, Stage, FORMAT};

pub fn shot(args: &[String]) -> Res<()> {
    let scene = args.first().ok_or("shot: missing scene id")?.clone();
    let out_path = args.get(1).filter(|a| !a.starts_with("--")).ok_or("shot: missing output path (it comes before the options)")?.clone();
    let mut options = Options::new(&args[2..]);
    let [w, h] = options.numbers("--size")?.unwrap_or([1600.0, 1000.0]).map(|v| (v as u32).max(1));
    let frames = options.parsed("--frames", 90u32)?.max(1);
    let scale: f32 = options.parsed("--scale", 1.0)?;
    if !(scale > 0.0 && scale.is_finite()) {
        return Err(format!("shot: --scale {scale} should be a number above 0").into());
    }
    let drags: Vec<[f32; 4]> = options.every("--drag")?;
    let scroll: Option<[f32; 3]> = options.numbers("--scroll")?;
    let clicks: Vec<[f32; 2]> = options.every("--click")?;
    // rest the pointer here once the clicks are done
    let hover: Option<[f32; 2]> = options.numbers("--hover")?;
    // print timings every `trace` frames (0: never)
    let trace: u32 = options.parsed("--trace", 0)?;
    // let time budgets apply, as they do in the window (the image then depends on the machine)
    let realtime = options.flag("--realtime");
    let panels = !options.flag("--no-panels");
    // only the picture: no panels, buttons or title
    let bare = options.flag("--bare");
    // (the scene's primer open over it)
    let primer = options.flag("--primer");
    options.done("shot")?;

    // Scripted input. Each action moves the pointer only in its own frames, so that
    // they can be combined: the clicks first, ten frames apart; then the scroll; then
    // the drags, one after another over the last two thirds of the run (each in a
    // share of that time: a frame to move to where it starts, the press, the movement,
    // the release, and a frame free before the next); and the pointer comes to rest
    // at `--hover` whenever nothing else has it.
    let clicks_end = 8 + 10 * clicks.len() as u32;
    const BETWEEN: u32 = 3;
    let first = (frames / 3).max(clicks_end + 3);
    let n = drags.len() as u32;
    let last = frames.saturating_sub(4).max(first + 2 * n + BETWEEN * n.saturating_sub(1));
    let share = (last - first + BETWEEN) / n.max(1);
    // (the frames in which each drag presses and lets go)
    let spans: Vec<(u32, u32)> = (0..n).map(|k| (first + k * share, first + k * share + share - BETWEEN)).collect();
    let needed = [
        // (the last click lets go in frame 7 + 10k)
        (10 * clicks.len() as u32).saturating_sub(2),
        if scroll.is_some() { clicks_end + 2 } else { 0 },
        spans.last().map_or(0, |&(_, end)| end + 1),
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    if needed > frames {
        return Err(format!("shot: the clicks, scroll and drags asked for need {needed} frames, and --frames is {frames}").into());
    }

    crate::clock::set_reproducible(!realtime);
    crate::ui::defer_unseen_text(realtime);
    let mut stage = Stage::showing(&scene)?;
    let (device, queue, format) = (stage.device.clone(), stage.queue.clone(), FORMAT);
    let app = &mut stage.app;
    app.set_panels(panels);
    if bare {
        app.set_bare();
    }
    if primer {
        app.show_primer(true);
    }

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("screenshot"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w as f32 / scale, h as f32 / scale));
    let start = std::time::Instant::now();
    let mut pass_time = 0.0f64;

    for frame in 0..frames {
        let mut raw =
            egui::RawInput { screen_rect: Some(screen), time: Some(frame as f64 / 60.0), predicted_dt: 1.0 / 60.0, focused: true, ..Default::default() };
        raw.viewports.entry(egui::ViewportId::ROOT).or_default().native_pixels_per_point = Some(scale);

        let button = |pos, pressed| egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: Default::default() };
        let mut busy = false;
        for (k, c) in clicks.iter().enumerate() {
            let at = 6 + 10 * k as u32;
            let pos = egui::pos2(c[0], c[1]);
            if frame + 2 == at || frame + 1 == at {
                raw.events.push(egui::Event::PointerMoved(pos));
            } else if frame == at {
                raw.events.push(button(pos, true));
            } else if frame == at + 1 {
                raw.events.push(button(pos, false));
            }
            busy |= frame + 2 >= at && frame <= at + 1;
        }
        if let Some([x, y, amount]) = scroll {
            if frame == clicks_end + 1 {
                busy = true;
                raw.events.push(egui::Event::PointerMoved(egui::pos2(x, y)));
                raw.events.push(egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: egui::vec2(0.0, amount),
                    phase: egui::TouchPhase::Move,
                    source: egui::MouseWheelSource::Wheel,
                    modifiers: Default::default(),
                });
            }
        }
        let mut dragged = false;
        for (&[x0, y0, x1, y1], &(begin, end)) in drags.iter().zip(&spans) {
            let (a, b) = (egui::pos2(x0, y0), egui::pos2(x1, y1));
            if frame + 1 == begin {
                raw.events.push(egui::Event::PointerMoved(a));
            } else if frame == begin {
                raw.events.push(egui::Event::PointerMoved(a));
                raw.events.push(button(a, true));
            } else if frame > begin && frame < end {
                let t = (frame - begin) as f32 / (end - 1 - begin).max(1) as f32;
                raw.events.push(egui::Event::PointerMoved(a + (b - a) * t.min(1.0)));
            } else if frame == end {
                raw.events.push(button(b, false));
            }
            busy |= frame + 1 >= begin && frame <= end;
            dragged |= frame > end;
        }
        if let Some([x, y]) = hover {
            // (after a drag the pointer stays where the drag let go of it)
            if !busy && !dragged && frame > clicks_end {
                raw.events.push(egui::Event::PointerMoved(egui::pos2(x, y)));
            }
        }

        let pass_start = std::time::Instant::now();
        let out = stage.frame(raw);
        let (app, ctx, renderer) = (&mut stage.app, &stage.ctx, &stage.renderer);
        // what a real frame also pays: turning the interface into triangles
        if trace > 0 {
            let _ = ctx.tessellate(out.shapes.clone(), out.pixels_per_point);
        }
        pass_time += pass_start.elapsed().as_secs_f64();
        if trace > 0 && (frame + 1) % trace == 0 {
            let n = app.frames.max(1) as f64;
            let (segs, pts, tris) = app.stats();
            let (drawn, reused) = app.frame_counts();
            println!(
                "frame {:>6}   interface {:>6.3} ms   scene {:>7.3} ms   render {:>7.3} ms   whole pass {:>7.3} ms   {segs} segments  {pts} points  {tris} triangles   drawn {drawn} reused {reused}",
                frame + 1,
                (app.total_cpu - app.scene_cpu - app.render_cpu) / n * 1e3,
                app.scene_cpu / n * 1e3,
                app.render_cpu / n * 1e3,
                pass_time / trace as f64 * 1e3
            );
            pass_time = 0.0;
            app.reset_timers();
        }
        if frame + 1 == frames {
            let prims = ctx.tessellate(out.shapes, out.pixels_per_point);
            let sd = egui_wgpu::ScreenDescriptor { size_in_pixels: [w, h], pixels_per_point: scale };
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("egui") });
            let extra = renderer.write().update_buffers(&device, &queue, &mut enc, &prims, &sd);
            {
                let mut rp = enc
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("egui"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.03, g: 0.035, b: 0.05, a: 1.0 }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    })
                    .forget_lifetime();
                renderer.read().render(&mut rp, &prims, &sd);
            }
            queue.submit(extra.into_iter().chain([enc.finish()]));
        }
        stage.done(&out.textures_delta);
    }

    // read the image back
    let row = (w * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: row as u64 * h as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: &target, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit([enc.finish()]);
    buf.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
    device.poll(wgpu::PollType::wait_indefinitely())?;
    let data = buf.slice(..).get_mapped_range()?;
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h as usize {
        let s = y * row as usize;
        pixels.extend_from_slice(&data[s..s + (w * 4) as usize]);
    }
    drop(data);

    let file = std::io::BufWriter::new(std::fs::File::create(&out_path)?);
    let mut encoder = png::Encoder::new(file, w, h);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&pixels)?;
    eprintln!("{scene}: {frames} frames at {w}x{h} in {:.1}s -> {out_path}", start.elapsed().as_secs_f32());
    Ok(())
}
