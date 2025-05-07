# Monitoring Stack Deployment

This directory contains the Kubernetes configuration for deploying the Prometheus and Grafana monitoring stack using the `kube-prometheus-stack` Helm chart.

## Prerequisites

*   Helm v3 installed (`helm version`)
*   Kubectl configured to your Kubernetes cluster (`kubectl cluster-info`)
*   A StorageClass available in your cluster for persistence (check with `kubectl get storageclass`). If your default isn't suitable, update `prometheus-values.yaml`.

## Deployment Steps

1.  **Add Helm Repository:**
    ```bash
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    ```

2.  **Install/Upgrade Monitoring Stack:**
    Deploy Prometheus, Grafana, Alertmanager, and associated components using the Helm chart and our custom values.
    ```bash
    # Choose a release name (e.g., "monitoring") and the namespace (e.g., "monitoring" or "default")
    helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
      --namespace default \
      --create-namespace \
      -f devops/k8s/monitoring/prometheus-values.yaml
    ```
    *(Note: Replace `--namespace default` if you prefer a dedicated 'monitoring' namespace)*

3.  **Apply ServiceMonitors:**
    Apply the ServiceMonitor configuration to tell Prometheus how to scrape your application(s).
    ```bash
    kubectl apply -f devops/k8s/monitoring/servicemonitor-fastapi.yaml --namespace default
    # Add other ServiceMonitors here if needed for other services
    ```
    *(Note: Ensure the namespace matches where you deployed the monitoring stack and your application)*

## Accessing Services

By default, the services are exposed within the cluster. Use `kubectl port-forward` to access them locally:

*   **Prometheus:**
    ```bash
    kubectl port-forward svc/monitoring-kube-prometheus-prometheus 9090:9090 --namespace default
    ```
    Access via: `http://localhost:9090`

*   **Grafana:**
    ```bash
    kubectl port-forward svc/monitoring-grafana 3000:80 --namespace default
    ```
    Access via: `http://localhost:3000`
    *(Default login: admin / prom-operator - Check chart values or secrets if changed)*

*   **Alertmanager:**
    ```bash
    kubectl port-forward svc/monitoring-kube-prometheus-alertmanager 9093:9093 --namespace default
    ```
    Access via: `http://localhost:9093`

*(Note: Replace `monitoring` in service names if you used a different Helm release name. Replace `--namespace default` if needed.)*

## Dashboards

*   Export Grafana dashboards as JSON and save them in the `grafana-dashboards/` directory.
*   Configure `prometheus-values.yaml` to automatically load these dashboards from ConfigMaps during deployment (see commented-out example in the values file).
