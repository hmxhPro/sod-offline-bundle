/**
 * src/pages/TrainingPage.jsx
 * --------------------------
 * The model-training workspace: left column (category manager + train panel +
 * trained-model list) and a right area that toggles between online annotation
 * and annotated-dataset import. State comes from ConsoleLayout via context.
 */

import React from 'react'
import { useOutletContext } from 'react-router-dom'
import { GraduationCap, MousePointer2, Database } from 'lucide-react'

import CategoryManager from '../components/CategoryManager'
import TrainPanel from '../components/TrainPanel'
import ModelListPanel from '../components/ModelListPanel'
import KonvaAnnotator from '../components/KonvaAnnotator'
import DatasetImporter from '../components/DatasetImporter'
import PageHeader from '../components/PageHeader'
import WorkspaceEmpty, { WorkspaceSteps } from '../components/WorkspaceEmpty'
import { TabButton } from '../components/ui'

export default function TrainingPage() {
  const {
    selectedCategory, setSelectedCategory, catReloadToken, modelReloadToken,
    trainTab, setTrainTab, refreshSelectedCategory, handleTrained,
  } = useOutletContext()

  return (
    <>
      <PageHeader
        title="模型训练"
        subtitle="将每一次标注沉淀为数据，训练适合具体场景的专用模型。"
        icon={<GraduationCap size={18} />}
      />

      <WorkspaceSteps steps={['管理类别', '标注与数据集', '训练专用模型', '投入检测']} />
      <div className="workspace-grid items-start">
        {/* ── Left: category / train / models ───────────────────── */}
        <aside className="flex flex-col gap-5">
          <CategoryManager
            selectedId={selectedCategory?.id ?? null}
            onSelect={setSelectedCategory}
            reloadToken={catReloadToken}
          />
          <TrainPanel category={selectedCategory} onTrained={handleTrained} />
          <ModelListPanel reloadToken={modelReloadToken} />
        </aside>

        {/* ── Right: annotate online OR import an annotated dataset ─── */}
        <section className="min-w-0">
          <div className="workspace-tabs flex items-center gap-2 mb-4">
            <TabButton
              active={trainTab === 'annotate'}
              onClick={() => setTrainTab('annotate')}
              icon={<MousePointer2 size={15} />}
            >
              在线标注
            </TabButton>
            <TabButton
              active={trainTab === 'import'}
              onClick={() => setTrainTab('import')}
              icon={<Database size={15} />}
            >
              上传数据集
            </TabButton>
          </div>

          {!selectedCategory ? (
            <WorkspaceEmpty
              label="数据工作区 / 模型训练"
              title="为你的场景，训练专用模型"
              description="先创建或选择一个类别，再进行在线标注或导入已有数据集。训练完成的模型可直接用于检测。"
              tags={['人工修正', '数据集导入', 'YOLO11 训练']}
            />
          ) : trainTab === 'annotate' ? (
            <KonvaAnnotator
              category={selectedCategory}
              onDataChanged={refreshSelectedCategory}
              initialTrainableOnly
            />
          ) : (
            <DatasetImporter category={selectedCategory} onDataChanged={refreshSelectedCategory} />
          )}
        </section>
      </div>
    </>
  )
}
