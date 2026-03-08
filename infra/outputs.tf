output "instance_id" {
  value = aws_instance.app.id
}

output "public_ip" {
  value = aws_instance.app.public_ip
}

output "streamlit_url" {
  value = "http://${aws_instance.app.public_ip}:${var.streamlit_port}"
}

output "api_url" {
  value = "http://${aws_instance.app.public_ip}:${var.api_port}"
}