"""WA-JEPA full-task VQ-BeT adaptation."""

VARIANT_ID = 'wa_vq_h5_a5_pixel_l1_v1'
HISTORY_OFFSETS = (-4, -3, -2, -1, 0)
ACTION_BLOCK = 5
PLAN_BLOCKS = 1
FUTURE_OFFSETS = (5,)


def experiment_config():
    return dict(
        architecture_variant=VARIANT_ID,
        history_offsets=list(HISTORY_OFFSETS),
        history_size=len(HISTORY_OFFSETS),
        plan_blocks=PLAN_BLOCKS,
        action_block=ACTION_BLOCK,
        prediction_steps=PLAN_BLOCKS * ACTION_BLOCK,
        future_offsets=list(FUTURE_OFFSETS),
        scene_tokens=8,
        embed_dim=192,
        refinement_stages=4,
        feedback_updates=3,
        action_loss='10 * pixel_L1 + 5 * focal_code_1 + 0.5 * focal_code_2',
        focal_gamma=2.0,
        action_tail_padding='repeat_last_action, supervised (Patch Policy convention)',
        action_stage_weights='final + 0.25 * mean(intermediate)',
        world_loss_weight=1.0,
        proprio_loss_weight=0.25,
        future_tail_mask=True,
    )
