variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "project_name" {
  type    = string
  default = "pft-poc"
}

variable "instance_type" {
  type    = string
  default = "m7i-flex.large"
}

variable "my_ip_cidr" {
  type        = string
  description = "Your public IP in CIDR form for SSH, e.g. 1.2.3.4/32"
}

variable "env_content" {
  type        = string
  sensitive   = true
  description = "Full contents of the .env file"
}

variable "streamlit_port" {
  type    = number
  default = 8501
}

variable "api_port" {
  type    = number
  default = 8000
}