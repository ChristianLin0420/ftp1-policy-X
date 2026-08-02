import dataclasses
import pathlib
from openpi.models import model as _model
import openpi.models.gemma as _gemma
from typing_extensions import override

#################### Important Definition for FTP1 Model #########################

# Defintion&Visualization: src/openpi/models_pytorch/definition_tactile_torque_function_area.png
FTP1_SINGLE_HAND_NUM_TACTILE_AREAS = 24

# Defintion&Visualization: src/openpi/models_pytorch/definition_faas_human_hand_joint.png
# TODO: add reference to UniDex
FTP1_SINGLE_HAND_JOINT_DIM = 32

FTP1_SINGLE_ARM_JOINT_DIM = 7

# Single arm action representation dimension = 9 (wrist 9dof) + (arm joints) + (hand joints) = 9 + 7 + 32 = 48
FTP1_SINGLE_ARM_ACTION_REP_DIM = 9 + FTP1_SINGLE_ARM_JOINT_DIM + FTP1_SINGLE_HAND_JOINT_DIM

# Reserved dimension for the future usage.
FTP1_RESERVED_ACTION_DIM = 15

# Total action dimension = 2 * (9dof-wrist-dim + arm-joint-dim + hand-dim) + 9 (9dof-head-dim) + reserved-dim = 2 * 48 + 9 + 15 = 120
# FTP1_ACTION_DIM = 2 * FTP1_SINGLE_ARM_ACTION_REP_DIM + 9 + FTP1_RESERVED_ACTION_DIM

FTP1_ACTION_HORIZON = 32

###################################################################################

# ${hand_side}_tactile_sensor_${name} --> T3-pretrained-checkpoints-url key (below).
T3_PRETRAINED_SENSOR_NAME_MAP = {
    "densetact": "densetact", "digit": "digit", "finray": "finray", "gs_black": "gs_black", "gs_tag": "gs_tag", "mini": "mini", "svelte": "svelte", "wedge": "wedge",
    "GelSightMini": "mini", "GelSightWedge": "wedge", "GelSightSvelte": "svelte", "GelSightFinray": "finray",
    "DIGIT": "digit",
    "DenseTact2": "densetact",
}

# the url to download the t3 pretrained checkpoint for the (hpt) tactile encoder.
# real url = base_url + T3_PRETRAINED_SENSOR_NAME_MAP[name] + ".pth"
T3_PRETRAINED_TACTILE_ENCODER_CHECKPOINTS_BASE_URL = "https://huggingface.co/datasets/alanz-mit/FoundationTactile/resolve/main/models/t3_large/encoders/"

###################################################################################

@dataclasses.dataclass(frozen=True)
class FTP1TactileTokenizerConfig:
    """Configuration for FTP1HptTactileEncoder."""

    single_hand_num_tactile_areas: int = FTP1_SINGLE_HAND_NUM_TACTILE_AREAS

    # [PACT] Tokens emitted per tactile function area.
    # 1  -> legacy FTP-1: the shared chunk trunk is reduced to its CLS token and the
    #       remaining 196 patch tokens are discarded (see SharedImageChunkEncoder).
    # >1 -> PACT spatial tokens: CLS ++ (spatial_pool_grid ** 2) adaptively pooled patch
    #       tokens, so a gel frame keeps spatial structure. tokens_per_area must equal
    #       1 + spatial_pool_grid ** 2.
    tokens_per_area: int = 1
    spatial_pool_grid: int = 3

    # [state encoding]: FourierStateEncoder parameters for vector-type tactile data
    fourier_dim: int = 8  # Dimension of Fourier encoding for each state dimension
    fourier_min_period: float = 1e-3  # Minimum period for Fourier encoding
    fourier_max_period: float = 1.0  # Maximum period for Fourier encoding
    
    # [matrix_encoding]: MatrixCNNEncoder parameters for matrix-type tactile data
    cnn_num_layers: int = 3  # Number of CNN layers for matrix encoding
    # Note: cnn_hidden_dim is automatically set to output_dim * 2 in MatrixCNNEncoder, Fourier coefficients shared with the state encoding.
    
    # [image_encoding]: Image-type tactile always uses the shared chunk encoder.
    frozen_shared_chunk: bool = False  # Whether to freeze shared chunk encoder parameters
    load_t3_pretrained_checkpoint: bool = True  # Whether to load t3 pretrained checkpoint for shared chunk encoder
    cache_t3_pretrained_checkpoint_dir: str = None  # If None, will be set to the assets directory of FTP1 / PI0 model.


@dataclasses.dataclass(frozen=True)
class FTP1ModelConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"
    tactile_expert_variant: _gemma.Variant = "gemma_small"  
    # Attention backend selection (forwarded to HF-style configs).
    # Values: "eager" (default), "sdpa", "flash_attention_2".
    attn_implementation: str = "eager"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 32
    max_token_len: int = None  # type: ignore
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    
    # FTP1's differences from Pi05:
    
    # - the state input is selected as: (1) not used, (2) add to adarms conditioning, (3) add to vlm backbones, (4) add to action expert backbones
    state_input_mode: str = 'action_expert' # 'none', 'adarms', 'vl_expert', 'action_expert'

    # - the hetegeneous tactile encoder & expert is added to the model.
    use_tactile_input: bool = True
    disable_history: bool = True

    # [PACT] Open the vision -> tactile attention edge. FTP-1 keeps the prefix and the
    # tactile branch strictly block-diagonal, so image and tactile tokens never attend to
    # each other at any layer; they meet only as independent K/V for the action expert.
    # When True, tactile rows may read the *image* columns of the prefix. Language columns
    # stay closed, and vision never reads tactile, so the prefix KV cache is preserved and
    # a fully gated batch is bit-identical to a vision-language-only policy.
    tactile_reads_vision: bool = False

    # [PACT] Drop the spatial tactile tokens of a function area that is not in contact,
    # keeping its CLS token. Requires the dataset to emit `tactile_contact`.
    contact_gating: bool = False
    tactile_input_config_file: str = None   # only for inference. When training, we generate it automatically from dataset.
    tactile_tokenizer_config: FTP1TactileTokenizerConfig = dataclasses.field(
        default_factory=FTP1TactileTokenizerConfig
    )
    # Assets directory for the model. Used to set default cache_t3_pretrained_checkpoint_dir if None.
    assets_dir: str | None = None

    def __post_init__(self):
        attn_impl = str(getattr(self, "attn_implementation", "eager"))
        allowed_attn_impl = {"eager", "sdpa", "flash_attention_2"}
        if attn_impl not in allowed_attn_impl:
            raise ValueError(
                f"Invalid attn_implementation: {attn_impl}. Expected one of {sorted(allowed_attn_impl)}."
            )

        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 150)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", False)
        if self.state_input_mode not in ['adarms', 'vl_expert', 'action_expert', 'none']:
            raise ValueError(f"Invalid state input mode: {self.state_input_mode}, should be one of ['adarms', 'vl_expert', 'action_expert', 'none']")

        tok_cfg = self.tactile_tokenizer_config
        expected_tokens = 1 + tok_cfg.spatial_pool_grid ** 2
        if tok_cfg.tokens_per_area not in (1, expected_tokens):
            raise ValueError(
                f"tokens_per_area={tok_cfg.tokens_per_area} is inconsistent with "
                f"spatial_pool_grid={tok_cfg.spatial_pool_grid}; expected 1 (legacy CLS) "
                f"or {expected_tokens} (CLS + {tok_cfg.spatial_pool_grid}x{tok_cfg.spatial_pool_grid} pool)."
            )
        if self.contact_gating and tok_cfg.tokens_per_area == 1:
            raise ValueError(
                "contact_gating requires tokens_per_area > 1: with a single CLS token per area "
                "there are no spatial tokens to gate, and dropping the CLS would remove the "
                "'pad present but feels nothing' signal."
            )

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.FTP1

    def set_single_hand_num_tactile_areas(self, single_hand_num_tactile_areas: int):
        object.__setattr__(self, 'tactile_tokenizer_config', dataclasses.replace(self.tactile_tokenizer_config, single_hand_num_tactile_areas=single_hand_num_tactile_areas))

    # We erase inputs_spec & get_freeze_filter from the base model config, since the FTP1 model only supports pytorch training & inference now.
    @override
    def create(self, rng) -> "FTP1":
        pass

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        pass

    def get_freeze_filter(self):
        pass
    
    def finalize_config(self, assets_dir: str | None = None):
        """Finalize the config after command line arguments are applied.
        
        Args:
            assets_dir: Assets directory to use. If None, uses self.assets_dir if set.
        """
        # Set cache_t3_pretrained_checkpoint_dir if None and assets_dir is provided
        if assets_dir is None:
            assets_dir = self.assets_dir
        
        if (self.tactile_tokenizer_config.cache_t3_pretrained_checkpoint_dir is None and 
            assets_dir is not None):
            cache_dir = pathlib.Path(assets_dir) / "checkpoints_t3_encoder_base"
            new_tokenizer_config = dataclasses.replace(
                self.tactile_tokenizer_config,
                cache_t3_pretrained_checkpoint_dir=str(cache_dir)
            )
            object.__setattr__(self, 'tactile_tokenizer_config', new_tokenizer_config)
