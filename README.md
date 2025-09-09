![alt](https://github.com/DanChitwood/ECT_animation/blob/main/outputs/leaf_morphospace_animation.gif)

From a saved PCA morphospace in `./data/saved_leaf_model_data/`, the script `./scripts/generate_animation.py` fits an ellipse in multidimensional PC space from which are continuously sampled theoretical grapevine eigenleaves and animation panels showing Cartesian and radial forms of the Euler Characteristic Transform (ECT) that are saved in `outputs`. To create `./outputs/leaf_morphospace_animation.mp4` and `./outputs/leaf_morphospace_animation.gif`, the following commands were used:

```bash
ffmpeg -framerate 24 -i "../outputs/animation_panels/frame_%03d.png" -c:v libx264 -pix_fmt yuv420p -movflags +faststart ../outputs/leaf_morphospace_animation.mp4
```

```bash
ffmpeg -i "../outputs/animation_panels/frame_%03d.png" -vf "fps=15,scale=512:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 ../outputs/leaf_morphospace_animation.gif
```
