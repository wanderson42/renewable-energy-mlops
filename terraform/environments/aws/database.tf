resource "aws_db_subnet_group" "mlflow" {
  name       = "${local.name}-mlflow"
  subnet_ids = [for subnet in aws_subnet.database : subnet.id]
}

resource "aws_security_group" "database" {
  name_prefix = "${local.name}-db-"
  description = "PostgreSQL from EKS workloads only"
  vpc_id      = aws_vpc.mlops.id
}

resource "aws_vpc_security_group_ingress_rule" "postgres" {
  security_group_id            = aws_security_group.database.id
  referenced_security_group_id = aws_eks_cluster.mlops.vpc_config[0].cluster_security_group_id
  ip_protocol                  = "tcp"
  from_port                    = 5432
  to_port                      = 5432
  description                  = "EKS managed nodes and pods to MLflow PostgreSQL"
}

resource "aws_db_instance" "mlflow" {
  identifier                      = "${local.name}-mlflow"
  engine                          = "postgres"
  engine_version                  = var.db_engine_version
  instance_class                  = "db.t4g.small"
  allocated_storage               = 20
  max_allocated_storage           = 100
  storage_type                    = "gp3"
  storage_encrypted               = true
  db_name                         = "mlflow"
  username                        = "mlflow_admin"
  manage_master_user_password     = true
  port                            = 5432
  db_subnet_group_name            = aws_db_subnet_group.mlflow.name
  vpc_security_group_ids          = [aws_security_group.database.id]
  publicly_accessible             = false
  multi_az                        = true
  backup_retention_period         = 7
  copy_tags_to_snapshot           = true
  deletion_protection             = true
  skip_final_snapshot             = false
  final_snapshot_identifier       = var.db_final_snapshot_identifier
  auto_minor_version_upgrade      = false
  apply_immediately               = false
  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]
}
