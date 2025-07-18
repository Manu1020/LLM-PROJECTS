FROM jenkins/jenkins:lts

USER root

RUN apt-get update -y && \
    apt-get install -y ca-certificates curl gnupg lsb-release && \
    curl -fsSL https://get.docker.com | sh

RUN groupadd -f docker && \
    usermod -aG docker jenkins

USER jenkins