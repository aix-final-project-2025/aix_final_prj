javascriptfunction runAnalysis(type) {
    $.ajax({
        url: `/run_analysis/${type}/`,
        method: 'GET',
        success: function(response) {
            let content = '';
            if (response.status === 'success') {
                if (type === 'preprocess') {
                    content = `<p>${response.message}</p><p>Data Shape: ${response.data_shape}</p>`;
                } else if (type === 'regression') {
                    content = '<h3>회귀 모델 결과</h3>';
                    for (let model in response.results) {
                        content += `<p><strong>${model}</strong>:<br>`;
                        content += `RMSE: ${response.results[model].RMSE.toFixed(3)}<br>`;
                        content += `R²: ${response.results[model].R2.toFixed(3)}<br>`;
                        content += `MAE: ${response.results[model].MAE.toFixed(3)}<br>`;
                        content += `MAPE: ${response.results[model].MAPE.toFixed(1)}%<br>`;
                        content += `Improvement: ${response.results[model].improvement.toFixed(1)}%</p>`;
                        if (response.results[model].image) {
                            content += `<img src="data:image/png;base64,${response.results[model].image}" alt="${model} Feature Importance">`;
                        }
                    }
                } else if (type === 'classification') {
                    content = '<h3>분류 모델 결과</h3>';
                    for (let model in response.results) {
                        content += `<p><strong>${model}</strong>:<br>`;
                        content += `Accuracy: ${response.results[model].accuracy.toFixed(3)}<br>`;
                        content += `Precision: ${response.results[model].precision.toFixed(3)}<br>`;
                        content += `Recall: ${response.results[model].recall.toFixed(3)}<br>`;
                        content += `F1-Score: ${response.results[model].f1.toFixed(3)}<br>`;
                        content += `AUC: ${response.results[model].auc.toFixed(3)}</p>`;
                    }
                } else if (type === 'clustering') {
                    content = '<h3>군집 모델 결과</h3>';
                    content += `<p>Optimal K: ${response.results.optimal_k}</p>`;
                    content += `<p>Silhouette Score: ${response.results.silhouette_score.toFixed(3)}</p>`;
                    content += `<img src="data:image/png;base64,${response.results.pca_image}" alt="PCA Plot">`;
                    content += '<h4>Cluster Analysis</h4>';
                    for (let cluster in response.results.cluster_counts) {
                        content += `<p><strong>Cluster ${cluster}</strong> (n=${response.results.cluster_counts[cluster]}):<br>`;
                        for (let feature in response.results.cluster_analysis[cluster]) {
                            content += `${feature}: ${response.results.cluster_analysis[cluster][feature].toFixed(2)}<br>`;
                        }
                        content += '</p>';
                    }
                } else if (type === 'deep_learning') {
                    content = `<p>${response.message}</p>`;
                }
            } else {
                content = `<p>Error: ${response.message}</p>`;
            }
            $('#result-content').html(content);
        },
        error: function() {
            $('#result-content').html('<p>분석 중 오류가 발생했습니다.</p>');
        }
    });
}

$('#prediction-form').submit(function(e) {
    e.preventDefault();
    $.ajax({
        url: '/predict_deep_learning/',
        method: 'POST',
        data: $(this).serialize(),
        success: function(response) {
            let content = '';
            if (response.status === 'success') {
                content = `<h3>딥러닝 예측 결과</h3>`;
                content += `<p>Predicted Coffee Intake Level: ${response.predicted_class}</p>`;
                content += `<p>Probabilities:</p>`;
                response.probabilities.forEach((prob, index) => {
                    content += `<p>Level ${index}: ${(prob * 100).toFixed(2)}%</p>`;
                });
            } else {
                content = `<p>Error: ${response.message}</p>`;
            }
            $('#result-content').html(content);
        },
        error: function() {
            $('#result-content').html('<p>예측 중 오류가 발생했습니다.</p>');
        }
    });
});