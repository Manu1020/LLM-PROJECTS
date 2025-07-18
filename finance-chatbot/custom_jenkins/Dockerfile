FROM jenkins/jenkins:lts

USER root

RUN apt-get update -y && \
    apt-get install -y ca-certificates curl gnupg lsb-release && \
    curl -fsSL https://get.docker.com | sh

RUN groupadd -f docker && \
    usermod -aG docker jenkins

USER jenkins


# docker exec -u root -it jenkins-dind bash
# apt update
# apt install -y
# apt install -y unzip curl
## Install Trivy
# curl -LO https://github.com/aquasecurity/trivy/releases/download/v0.62.1/trivy_0.62.1_Linux-ARM64.tar.gz
# tar zxvf trivy_0.62.1_Linux-ARM64.tar.gz
# mv trivy /usr/local/bin/

## Install awscli
# curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
# unzip awscliv2.zip
# ./aws/install
# aws --version
# exit


