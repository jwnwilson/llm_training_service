provider "aws" {
  region = var.aws_region
}

module "ecr" {
  source                = "./modules/ecr"
  repo_name             = var.repo_name
  image_retention_count = var.image_retention_count
}

module "ecr_temporal_ui" {
  source                = "./modules/ecr"
  repo_name             = "llm-api-temporal-ui"
  image_retention_count = var.image_retention_count
}

module "ecr_proxy" {
  source                = "./modules/ecr"
  repo_name             = "llm-api-proxy"
  image_retention_count = var.image_retention_count
}

module "ecr_inference" {
  source                = "./modules/ecr"
  repo_name             = "llm-api-inference"
  image_retention_count = var.image_retention_count
}

module "ecr_training" {
  source                = "./modules/ecr"
  repo_name             = "llm-api-training"
  image_retention_count = var.image_retention_count
}

module "ecr_export" {
  source                = "./modules/ecr"
  repo_name             = "llm-api-export"
  image_retention_count = var.image_retention_count
}

# Public ECR repository for the RunPod training image (AMD64 + CUDA).
# ECR Public allows RunPod to pull without credentials or token refresh.
# Must stay in us-east-1 — ECR Public's API endpoint is region-locked there.
module "ecr_public_runpod" {
  source    = "./modules/ecr_public"
  repo_name = "llm-api-runpod"
}

module "acm_ui" {
  source = "./modules/acm"
  domain = "llm.jwnwilson.co.uk"
}

module "s3_ui" {
  source              = "./modules/s3_static"
  name                = "llm-api-ui"
  domain              = "llm.jwnwilson.co.uk"
  acm_certificate_arn = module.acm_ui.certificate_arn
}

module "iam" {
  source                     = "./modules/iam"
  repo_name                  = var.repo_name
  github_repo                = var.github_repo
  s3_bucket                  = var.s3_bucket
  ecr_push_policy_arn        = module.ecr.ecr_push_policy_arn
  extra_ecr_push_policy_arns = [
    module.ecr_temporal_ui.ecr_push_policy_arn,
    module.ecr_proxy.ecr_push_policy_arn,
    module.ecr_inference.ecr_push_policy_arn,
    module.ecr_training.ecr_push_policy_arn,
    module.ecr_export.ecr_push_policy_arn,
    module.ecr_public_runpod.ecr_public_push_policy_arn,
  ]
  ecr_pull_repo_arns = [
    module.ecr.repository_arn,
    module.ecr_temporal_ui.repository_arn,
    module.ecr_proxy.repository_arn,
    module.ecr_inference.repository_arn,
    module.ecr_training.repository_arn,
    module.ecr_export.repository_arn,
  ]
  ui_bucket_arn              = module.s3_ui.bucket_arn
  ui_distribution_arn        = module.s3_ui.distribution_arn
  create_ui_resources        = true
}

module "dns" {
  source       = "./modules/dns"
  vps_ip       = var.vps_ip
  ui_cf_domain     = module.s3_ui.cloudfront_domain
  create_ui_record = true
}

# ── ML artifact bucket lifecycle — reduces storage costs ──────────────────
# The bucket was created manually; use a data source to attach lifecycle rules
# without importing or recreating it.
data "aws_s3_bucket" "ml_artifacts" {
  bucket = var.s3_bucket
}

# Versioning turns accidental deletes and lifecycle expiry into recoverable
# delete markers. Old versions are cleaned up by the noncurrent rule below.
resource "aws_s3_bucket_versioning" "ml_artifacts" {
  bucket = data.aws_s3_bucket.ml_artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Finished models live under model/ and datasets under dataset/ — neither may
# ever expire. tests/unit/test_storage_paths.py enforces this for model keys.
resource "aws_s3_bucket_lifecycle_configuration" "ml_artifacts" {
  bucket = data.aws_s3_bucket.ml_artifacts.id

  # Versioning must be enabled before noncurrent-version rules take effect.
  depends_on = [aws_s3_bucket_versioning.ml_artifacts]

  # DISABLED: until the run_model_key change (model/runs/{run_id}/) is deployed,
  # production still writes finished models to workflow/{run_id}/model/.
  # This rule silently deleted every pipeline-trained model in 2026.
  # Re-enable only once no model or dataset can be stored under workflow/.
  rule {
    id     = "expire-workflow-artifacts"
    status = "Disabled"
    filter { prefix = "workflow/" }
    expiration { days = 30 }
  }

  rule {
    id     = "expire-noncurrent-versions"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration { noncurrent_days = 7 }
    expiration { expired_object_delete_marker = true }
  }

  rule {
    id     = "abort-incomplete-multipart"
    status = "Enabled"
    filter {}
    abort_incomplete_multipart_upload { days_after_initiation = 3 }
  }
}
