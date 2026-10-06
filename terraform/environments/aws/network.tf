locals {
  name = "${var.project}-${var.environment}"
  zones = {
    for index, zone in var.availability_zones : tostring(index) => zone
  }
}

resource "aws_vpc" "mlops" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = { Name = local.name }
}

resource "aws_internet_gateway" "egress" {
  vpc_id = aws_vpc.mlops.id
}

resource "aws_subnet" "public" {
  for_each                = local.zones
  vpc_id                  = aws_vpc.mlops.id
  availability_zone       = each.value
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, tonumber(each.key))
  map_public_ip_on_launch = false
  tags                    = { Name = "${local.name}-egress-${each.key}" }
}

resource "aws_subnet" "workloads" {
  for_each                = local.zones
  vpc_id                  = aws_vpc.mlops.id
  availability_zone       = each.value
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, 10 + tonumber(each.key))
  map_public_ip_on_launch = false
  tags = {
    Name                              = "${local.name}-workloads-${each.key}"
    "kubernetes.io/role/internal-elb" = "1"
  }
}

resource "aws_subnet" "database" {
  for_each                = local.zones
  vpc_id                  = aws_vpc.mlops.id
  availability_zone       = each.value
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, 20 + tonumber(each.key))
  map_public_ip_on_launch = false
  tags                    = { Name = "${local.name}-database-${each.key}" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.mlops.id
}

resource "aws_route" "internet" {
  route_table_id         = aws_route_table.public.id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.egress.id
}

resource "aws_route_table_association" "public" {
  for_each       = local.zones
  subnet_id      = aws_subnet.public[each.key].id
  route_table_id = aws_route_table.public.id
}

resource "aws_eip" "nat" {
  for_each = local.zones
  domain   = "vpc"
}

resource "aws_nat_gateway" "egress" {
  for_each      = local.zones
  allocation_id = aws_eip.nat[each.key].id
  subnet_id     = aws_subnet.public[each.key].id
  depends_on    = [aws_internet_gateway.egress]
}

resource "aws_route_table" "workloads" {
  for_each = local.zones
  vpc_id   = aws_vpc.mlops.id
}

resource "aws_route" "workload_egress" {
  for_each               = local.zones
  route_table_id         = aws_route_table.workloads[each.key].id
  destination_cidr_block = "0.0.0.0/0"
  nat_gateway_id         = aws_nat_gateway.egress[each.key].id
}

resource "aws_route_table_association" "workloads" {
  for_each       = local.zones
  subnet_id      = aws_subnet.workloads[each.key].id
  route_table_id = aws_route_table.workloads[each.key].id
}

# Database subnets have only the VPC-local route, with no NAT or internet route.
resource "aws_route_table" "database" {
  vpc_id = aws_vpc.mlops.id
  route  = []
}

resource "aws_route_table_association" "database" {
  for_each       = local.zones
  subnet_id      = aws_subnet.database[each.key].id
  route_table_id = aws_route_table.database.id
}

resource "aws_vpc_endpoint" "s3" {
  vpc_id            = aws_vpc.mlops.id
  service_name      = "com.amazonaws.${var.region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [for table in aws_route_table.workloads : table.id]
}
