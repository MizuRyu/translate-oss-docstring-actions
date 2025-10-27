## actionsの仕様
- userは、あらゆる設定値に加えて、対象のリポジトリURL（repositoryが複数の言語のものレポである場合はサブディレクトリのURL）を渡すことができる
それをcheckout したソースにcloneして準備。cloneしたものを今回で言うtest_dataみたいな扱いとする

- userは、LLM実行前に、cloneしたソースコードからextractした時点で、この先実行するかhuman in the loopすることができる。userが承認したらLLM処理を始める。この時の承認時には、input tokenの全量がどの程度あるのかをuser側にフィードバックする。それに応じて、実行するかしないかを決めることができる

- 現状存在しているmockモードとは別にtest modeみたいなものも欲しい。test modeはLLMの最大実行回数をinputで決めて実行する形式。これにより暴発を防ぎ、一旦テストで通してみたい、というニーズに応えられるようにして欲しい

- プロセス内でつくられる　JSONLファイルは、ログの記録（何件あるとか。）だけ返し、実ファイルはtemporaryとしてcheckout したvmに残す

- 各プロセスはJSONLでやり取りする

- output dirやinput dirは、CLI実行ではuserが決めているが、GitHub actionsのworkflowで実行するときにはactions側に任せるようにする。

- direct modeとかindirect modeとかはactionsでは特にどうでもよくてuserがactionsを設置している、repositoryに翻訳結果を展開する。
- 途中失敗やunprocessed.jsonlがある場合は、途中まで成功した分を一旦repositoryに展開し、失敗したものはrepositoryにunprocessed.jsonlを作っちゃって残りを実行するpipelineに任せる


### 開発容易性
act と言うものを使い、github actionsをローカルで実行できるようにしたい。
（私はmacbookを使用している。）

### 今後
失敗したjsonlレコードを実行するためのactions（workflow）が必要。
リカバリー用として。

できれば、test用と実際のtranslate.ymlは分けたいところ。

今後は、更新差分があればその差分を検知（commitベースで）して、そのドキュメンテーション化も同期するようにしてきたいと思っている。
